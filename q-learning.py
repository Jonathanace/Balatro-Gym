# https://gymnasium.farama.org/introduction/train_agent/
from collections import defaultdict
import numpy as np
from main import BalatroEnv
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt


class BalatroQLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[obs][action]
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def _train_q_learning_agent():
    # WARNING: This does not work.
    learning_rate = 0.01
    n_episodes = 100
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = gym.make("Balatro-v0")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BalatroQLearningAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs
        agent.decay_epsilon()

    rolling_length = 500
    fig, axs = plt.subplot(ncols=3, figsize=(12, 5))

    axs[0].set_title("Episode Rewards")
    reward_moving_average = _get_moving_avgs(
        env.return_queue,
        window=rolling_length,
        convlution_mode="valid",
    )

    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    axs[1].set_title("Episode Lengths")
    length_moving_average = _get_moving_avgs(
        env.length_queue,
        window=rolling_length,
        convlution_mode="valid",
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    axs[2].set_title("Training Error")
    training_error_moving_average = _get_moving_avgs(
        agent.training_error,
        window=rolling_length,
        convlution_mode="same",
    )
    axs[2].plot(
        range(len(training_error_moving_average)), training_error_moving_average
    )
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout
    plt.show()


def _get_moving_avgs(arr, window, convlution_mode):
    return (
        np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convlution_mode,
        )
        / window
    )


def _manual_test():
    env = gym.make("Balatro-v0")
    obs, info = env.reset(seed=42)

    action = env.action_space.sample()
    action["action_type"] = 0 # Skip or Select Blind
    action["action args"]["skip_or_select_blind"] = 1  # Skip Blind
    env.step(action)

    action = env.action_space.sample()
    action["action_type"] = 1 # Play Hand or Discard
    action["action args"]["play_hand_or_discard"] = {
        "action": 0,  # Play Hand
        "cards": [1, 0, 1, 0, 0],  # Play cards at index 0 and 2
    }
    env.step(action)


if __name__ == "__main__":
    _train_q_learning_agent()
