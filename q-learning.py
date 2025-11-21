from main import BalatroEnv
import gymnasium as gym


def train_q_learning_agent():
    return


def manual_test():
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
    manual_test()
