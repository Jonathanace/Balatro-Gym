from mini_env import register_mini_env
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
from torchrl.modules import ProbabilisticActor, OneHotCategorical
from torchrl.envs.utils import set_exploration_type, ExplorationType

# Custom imports
from mini_env import register_mini_env
from main import MAX_HELD_HAND_SIZE
from wrappers import CombinatorialActionWrapper
from train_ppo import OneHotObservationWrapper

# ================= CONFIGURATION =================
CHECKPOINT_PATH = "ppo_balatro_mini_env_checkpoint.pt"
NUM_GAMES = 100  # How many games to "play"
# =================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

register_mini_env()
raw_env = gym.make("Balatro-Blind-Mini-v0")
raw_env = CombinatorialActionWrapper(raw_env, max_held_cards=MAX_HELD_HAND_SIZE)

wrapped_env = OneHotObservationWrapper(raw_env)

base_env = GymWrapper(wrapped_env, device=device)

env = TransformedEnv(
    base_env,
    Compose(
        # DTypeCast is removed because OneHotObservationWrapper outputs Float32
        StepCounter(),
    ),
)

# ==============================================================================
# DEFINE MODEL ARCHITECTURE (matches training exactly)
# ==============================================================================
# Perform a dummy rollout to get the exact input shape
dummy_rollout = env.rollout(2)
obs_shape = dummy_rollout["observation"].shape[-1]
n_actions = env.action_spec.space.n
num_cells = 256

print(f"Detected Input Shape: {obs_shape}")

# Exact same architecture as training script
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

# ==============================================================================
# LOAD CHECKPOINT
# ==============================================================================
print(f"Loading brain from {CHECKPOINT_PATH}...")
try:
    loaded_checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy_module.load_state_dict(loaded_checkpoint["policy_state_dict"])
    print("Brain loaded successfully.")
except FileNotFoundError:
    print(f"Checkpoint {CHECKPOINT_PATH} not found!")
    exit()
except RuntimeError as e:
    print(f"Error loading state dict: {e}")
    print("Ensure the architecture configuration matches the training script exactly.")
    exit()

# ==============================================================================
# THE TOURNAMENT LOOP
# ==============================================================================
print(f"\n--- STARTING {NUM_GAMES} GAME TOURNAMENT (DETERMINISTIC) ---")

wins = 0
losses = 0
total_rewards = []

with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():

    for game in range(NUM_GAMES):
        # Reset env
        tensordict = env.reset()
        tensordict = tensordict.to(device)

        game_reward = 0
        done = False

        while not done:
            # 1. Ask the bot for a move (Deterministic)
            policy_module(tensordict)

            # 2. Step the environment
            tensordict = env.step(tensordict)

            # 3. Track Reward
            reward = tensordict["next", "reward"].item()
            game_reward += reward

            # 4. Check if done
            done = tensordict["next", "done"].item()

            # 5. Prepare for next step
            tensordict = tensordict["next"]

        total_rewards.append(game_reward)

        # Win condition check
        if game_reward > 0:
            wins += 1
            result = "WIN"
        else:
            losses += 1
            result = "LOSS"

        print(f"Game {game+1}: {result} (Score: {game_reward:.2f})")

# ==============================================================================
# FINAL STATS
# ==============================================================================
win_rate = (wins / NUM_GAMES) * 100
avg_score = np.mean(total_rewards)

print("\n--- TOURNAMENT RESULTS ---")
print(f"Games Played: {NUM_GAMES}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Average Reward: {avg_score:.4f}")

if win_rate > 95:
    print("VERDICT: The bot has solved the seed.")
elif win_rate > 50:
    print("VERDICT: The bot is winning, but sometimes makes mistakes.")
else:
    print("VERDICT: The bot is still losing.")
