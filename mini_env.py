import gymnasium as gym
import logging
import sys
from gymnasium.utils.env_checker import check_env
from typing import Optional
from balatrobot import BalatroClient
import numpy as np
from enum import IntEnum
from main import MAX_HELD_HAND_SIZE, _build_map, CARD_RANKS, CARD_SUITS, _output_key_diff, save_data
from gymnasium.spaces import Dict, Box, Discrete, MultiBinary
from typing import NamedTuple
import balatrobot.enums


# Start with just "cards in hand"

class CardFeature(NamedTuple):
    name: str
    low: int
    high: int

HAND_SCHEMA = [
    CardFeature("rank", 0, 13),
    CardFeature("suit", 0, 4),
]

HandIdx = IntEnum("HandIdx", {f.name.upper(): i for i, f in enumerate(HAND_SCHEMA)})

class BalatroEnv(gym.Env):
    RANK_MAP = _build_map(CARD_RANKS, start_index=1)
    SUIT_MAP = _build_map(CARD_SUITS, start_index=1)

    def __init__(self, deck: str = "Red Deck", stake: int = 1):
        self.client = BalatroClient()
        self.client.connect()
        self.action_space = []

        self.deck = deck
        self.stake = stake

        hand_feat_lows = [f.low for f in HAND_SCHEMA]
        hand_feat_highs = [f.high for f in HAND_SCHEMA]

        self.observation_space = Dict(
            {
                "hand": Box(
                    low=np.tile(hand_feat_lows, (MAX_HELD_HAND_SIZE, 1)),
                    high=np.tile(   hand_feat_highs, (MAX_HELD_HAND_SIZE, 1)),
                    shape=(MAX_HELD_HAND_SIZE, len(HAND_SCHEMA)),
                    dtype=np.int32,
                ),
                "state": Discrete(len(balatrobot.enums.State)),
            }
        )

        self.action_space = Dict({
            "play_or_discard": Discrete(2),
            "target_cards": MultiBinary(MAX_HELD_HAND_SIZE),
        })

    def __del__(self):
        if getattr(self, "client", None):
            self.client.disconnect()

    def close(self):
        if getattr(self, "client", None):
            self.client.disconnect()

    def _get_obs(self):
        game_obs = self.client.send_message("get_game_state", {})
        cards_in_hand = game_obs["hand"]["cards"]

        current_hand = np.zeros(
                (MAX_HELD_HAND_SIZE, len(HAND_SCHEMA)), dtype=np.int32
            )

        for i, card in enumerate(cards_in_hand):
            card_config = card["config"]["card"]
            current_hand[i, HandIdx.RANK] = self.RANK_MAP[card_config["value"]]
            current_hand[i, HandIdx.SUIT] = self.SUIT_MAP[card_config["suit"]]

        current_obs = {
            "hand": current_hand,
            "state": game_obs["state"]
        }

        return current_obs

    def _start_game(self, seed: str = "0000155"):
        self.seed = seed
        self.client.send_message("go_to_menu", {})
        self.client.send_message(
            "start_run",
            {
                "deck": self.deck,
                "stake": self.stake,
                "seed": self.seed
            }
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._start_game()
        self.client.send_message("skip_or_select_blind", {"action": "select"})

        observation = self._get_obs()
        info = self.get_info()

        return observation, info

    def get_info(self):
        return {}

    def step(self, action):
        action_arg = "play" if action["play_or_discard"] == 0 else "discard"

        target_card_indices = [
            i for i, selected in enumerate(action["target_cards"]) if selected
        ]
        if len(target_card_indices) > 5:
            logging.error(f"{len(target_card_indices)} cards selected to {action_arg}!")
            reward_modifier = -1
        else:
            logging.info(f"Action: {action_arg} cards at indices {target_card_indices}")
            self.client.send_message(
                "play_hand_or_discard",
                {
                    "action": action_arg,
                    "cards": target_card_indices,
                }
            )

        observation = self._get_obs()
        state_id = observation["state"]
        game_state = balatrobot.enums.State(state_id).name
        if game_state == "GAME_OVER":
            reward = 0
            terminated = True
        elif game_state == "BLIND_SELECT":
            # won the blind
            reward = 1
            terminated = True
        else:
            reward = 0
            terminated = False

        return observation, reward + reward_modifier, terminated, False, {}

def register_mini_env():
    gym.register(
        id="Balatro-Blind-Mini-v0",
        entry_point="mini_env:BalatroEnv",
        max_episode_steps=10,
    )
if __name__ == "__main__":
    env = gym.make("Balatro-Blind-Mini-v0")
    obs, info = env.reset()
    _output_key_diff(env, obs)
    check_env(env)
    # save_data()
    # sys.exit(0)
