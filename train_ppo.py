import torch
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from torch import nn
from torchrl.envs import (
    Compose,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymWrapper
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator, OneHotCategorical
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from collections import defaultdict
from tqdm import tqdm
import os
import multiprocessing

# Import your custom modules
from mini_env import register_mini_env
from main import MAX_HELD_HAND_SIZE
from wrappers import CombinatorialActionWrapper


class OneHotObservationWrapper(gym.ObservationWrapper):
    """
    Standard Gym Wrapper to convert Integer Observations to One-Hot Floats.
    Runs BEFORE TorchRL sees the environment.
    """

    def __init__(self, env):
        super().__init__(env)

        # 1. Extract Limits from the original MultiDiscrete space
        # shape: (Cards * Features,) e.g., [15, 5, 15, 5...]
        self.hand_nvec = env.observation_space["hand"].nvec.flatten()
        self.state_n = int(env.observation_space["state"].n)

        # 2. Calculate new flat size
        self.out_dim = np.sum(self.hand_nvec) + self.state_n

        # 3. Define the new Observation Space
        # It is now a single Box of Floats (0.0 to 1.0)
        self.observation_space = spaces.Dict(
            {
                "observation": spaces.Box(
                    low=0.0, high=1.0, shape=(self.out_dim,), dtype=np.float32
                )
            }
        )
        print(f"OneHotWrapper: Converted Ints -> Box(Float32) size {self.out_dim}")

    def observation(self, obs):
        # 1. Get raw data
        hand_ints = obs["hand"].flatten()
        state_int = obs["state"]

        # 2. One-Hot Encode Hand
        encoded_parts = []
        for i, limit in enumerate(self.hand_nvec):
            val = hand_ints[i]
            # Create zero vector
            vec = np.zeros(limit, dtype=np.float32)
            # Set index to 1 (Safe clamp just in case)
            idx = min(max(0, val), limit - 1)
            vec[idx] = 1.0
            encoded_parts.append(vec)

        # 3. One-Hot Encode State
        state_vec = np.zeros(self.state_n, dtype=np.float32)
        idx = min(max(0, state_int), self.state_n - 1)
        state_vec[idx] = 1.0
        encoded_parts.append(state_vec)

        # 4. Concatenate
        final_obs = np.concatenate(encoded_parts)

        # Return as a Dict because TorchRL expects a Dict observation usually
        return {"observation": final_obs}


# ==============================================================================
# CONFIGURATION
# ==============================================================================
OUTPUT_PATH = "ppo_balatro_mini_env_checkpoint.pt"

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

num_cells = 256
lr = 5e-5
max_grad_norm = 0.5
frames_per_batch = 4096
total_frames = 500_000
sub_batch_size = 64
num_epochs = 4
clip_epsilon = 0.1
gamma = 0.99
lmbda = 0.95
entropy_eps = 0.05

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
register_mini_env()
raw_env = gym.make("Balatro-Blind-Mini-v0")
raw_env = CombinatorialActionWrapper(raw_env, max_held_cards=MAX_HELD_HAND_SIZE)

# Wrap the gym environment, convert integers to floats
wrapped_env = OneHotObservationWrapper(raw_env)

base_env = GymWrapper(wrapped_env, device=device)

# TorchRL Transform Pipeline
env = TransformedEnv(
    base_env,
    Compose(
        StepCounter(),
    ),
)

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================
dummy_rollout = env.rollout(2)
obs_shape = dummy_rollout["observation"].shape[-1]
print(f"Network Input Shape: {obs_shape} (Should be ~171)")

n_actions = env.action_spec.space.n

actor_net = nn.Sequential(
    nn.Linear(obs_shape, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, n_actions, device=device),
)

policy_module = TensorDictModule(
    actor_net, in_keys=["observation"], out_keys=["logits"]
)

policy_module = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec,
    in_keys=["logits"],
    distribution_class=OneHotCategorical,
    return_log_prob=True,
)

value_net = nn.Sequential(
    nn.Linear(obs_shape, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, num_cells, device=device),
    nn.Tanh(),
    nn.Linear(num_cells, 1, device=device),
)

value_module = ValueOperator(
    module=value_net,
    in_keys=["observation"],
)

# ==============================================================================
# TRAINING LOOP
# ==============================================================================
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch, device=device),
    sampler=SamplerWithoutReplacement(),
)

advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    average_gae=True,
    device=device,
)

loss_module = ClipPPOLoss(
    actor_network=policy_module,
    critic_network=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=bool(entropy_eps),
    entropy_coef=entropy_eps,
    critic_coef=1.0,
    loss_critic_type="smooth_l1",
)

optim = torch.optim.Adam(loss_module.parameters(), lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, total_frames // frames_per_batch, 0.0
)
logs = defaultdict(list)
pbar = tqdm(total=total_frames)

start_step = 0
if os.path.exists(OUTPUT_PATH):
    print(f"Loading checkpoint from {OUTPUT_PATH}...")
    loaded_checkpoint = torch.load(OUTPUT_PATH, map_location=device)
    policy_module.load_state_dict(loaded_checkpoint["policy_state_dict"])
    value_module.load_state_dict(loaded_checkpoint["value_state_dict"])
    optim.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    start_step = loaded_checkpoint.get("step", 0) + 1
    print("Resuming training!")

frames_remaining = total_frames - (start_step * frames_per_batch)
if frames_remaining <= 0:
    print("Training complete.")
    exit()

collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=frames_remaining,
    split_trajs=False,
    device=device,
)

pbar = tqdm(total=total_frames, initial=(start_step * frames_per_batch))


def save_checkpoint(current_step):
    torch.save(
        {
            "policy_state_dict": policy_module.state_dict(),
            "value_state_dict": value_module.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "step": current_step,
        },
        OUTPUT_PATH,
    )
    print(f"Checkpoint saved: {OUTPUT_PATH}")


try:
    for i, tensordict_data in enumerate(collector):
        # input("Press Enter to continue...")
        current_step = start_step + i
        tensordict_data = tensordict_data.to(device)

        for _ in range(num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

                with open("loss_log.csv", "a") as f:
                    f.write(
                        f"{current_step},{loss_vals['loss_objective'].item()},{loss_vals['loss_critic'].item()},{loss_vals['loss_entropy'].item()}\n"
                    )

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        pbar.set_description(
            f"Reward: {logs['reward'][-1]:.4f} | LR: {optim.param_groups[0]['lr']:.6f}"
        )
        scheduler.step()

        if current_step % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(1000, policy_module)
                print(f"Eval Reward: {eval_rollout['next', 'reward'].sum().item():.4f}")
            save_checkpoint(current_step)

finally:
    save_checkpoint(current_step)
