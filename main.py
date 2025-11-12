import logging
from balatrobot.client import BalatroClient
from balatrobot.exceptions import BalatroError
from typing import Optional
import numpy as np
import gymnasium as gym
import contextlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BalatroEnv(gym.Env):
    def __init__(self, deck: str = "Red Deck", stake: int = 1):
        self.deck = deck
        self.stake = stake
        self.client = BalatroClient()
        self.client.connect()

    def close(self):
        if self.client:
            self.client.disconnect()
            self.client = None


    def __del__(self):
        self.client.disconnect()

    ### Helper functions
    @staticmethod
    def balatro_function(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BalatroError as e:
                logger.error(f"API Error: {e}")
                logger.error(f"Error code: {e.error_code}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        return wrapper

    @balatro_function
    def _start_game(self,
                    deck: str = "Red Deck",
                    stake: int = 1,
                    seed: str = "0000155"):
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.client.send_message("go_to_menu", {})
        self.client.send_message(
            "start_run",
            {
                "deck": deck,
                "stake": stake,
                "seed": seed,
            }
        )


    ### Environment functions
    @balatro_function
    def _get_obs(self):
        return self.client.send_message("get_game_state", {})

    def _get_info(self):
        return {"deck": self.deck, "stake": self.stake, "seed": self.seed}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._start_game()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        info = self._get_info()



def test_run():
    env = BalatroEnv()
    try:
        obs, info = env.reset()
        obs = env._get_obs()
    finally:
        env.client.disconnect()

    print(obs)




if __name__ == "__main__":
    test_run()
