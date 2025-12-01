import gymnasium as gym
import logging
import sys
from gymnasium.utils.env_checker import check_env
from typing import Optional
from balatrobot import BalatroClient
import numpy as np
from enum import IntEnum
from main import MAX_HELD_HAND_SIZE, _build_map, CARD_RANKS, CARD_SUITS, _output_key_diff, save_data
from gymnasium.spaces import Dict, Box, Discrete, MultiBinary, MultiDiscrete
from typing import NamedTuple
import balatrobot.enums
from datetime import datetime


# Start with just "cards in hand"

class CardFeature(NamedTuple):
    name: str
    low: int
    high: int

HAND_SCHEMA = [
    CardFeature("rank", 0, 14),
    CardFeature("suit", 0, 5),
]


HandIdx = IntEnum("HandIdx", {f.name.upper(): i for i, f in enumerate(HAND_SCHEMA)})

class BalatroEnv(gym.Env):
    RANK_MAP = _build_map(CARD_RANKS, start_index=1)
    SUIT_MAP = _build_map(CARD_SUITS, start_index=1)

    def __init__(self, deck: str = "Red Deck", stake: int = 1):
        self.client = BalatroClient()
        self.client.connect()
        self.client._socket.settimeout(20.0)
        self.action_space = []
        self.current_chips = 0

        self.deck = deck
        self.stake = stake

        self.log_file = open("results.txt", "a")
        self.log("New Game Session")

        hand_feat_lows = [f.low for f in HAND_SCHEMA]
        hand_feat_highs = [f.high for f in HAND_SCHEMA]

        cards_per_hand = MAX_HELD_HAND_SIZE
        features_per_card = len(HAND_SCHEMA)

        hand_nvec = np.tile(hand_feat_highs, cards_per_hand)

        self.observation_space = Dict(
            {
                "hand": MultiDiscrete(hand_nvec),
                "state": Discrete(len(balatrobot.enums.State)),
            }
        )

        self.action_space = Dict({
            "play_or_discard": Discrete(2),
            "target_cards": MultiBinary(MAX_HELD_HAND_SIZE),
        })

    def log(self, message: str):
        self.log_file.write(f"{datetime.now().isoformat()}: {message}\n")
        self.log_file.flush()

    def __del__(self):
        if getattr(self, "client", None):
            self.client.disconnect()
        self.log_file.close()

    def close(self):
        if getattr(self, "client", None):
            self.client.disconnect()
        self.log_file.close()

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
        self.current_chips = 0

        return observation, info

    def get_info(self):
        return {}

    def step(self, action):
        full_obs = self.client.send_message("get_game_state", {})
        action_arg = "play_hand" if action["play_or_discard"] == 0 else "discard"

        blinds = full_obs["blinds"]
        current_blind = [b for b in blinds.values() if b["status"] == "Current"][0]
        chips_target = current_blind["score"]

        target_card_indices = [
            i for i, selected in enumerate(action["target_cards"]) if selected
        ]

        # Check for invalid plays
        invalid_action = False  # Assume action is valid by default
        state_id = full_obs["state"]
        game_state = balatrobot.enums.State(state_id).name
        if game_state != "SELECTING_HAND":
            pass
        elif len(target_card_indices) > 5 or len(target_card_indices) == 0:
            logging.error(f"{len(target_card_indices)} cards selected to {action_arg}!")
            invalid_action = True
        elif (
            action_arg == "discard"
            and full_obs["game"]["current_round"]["discards_left"] == 0
        ):
            logging.error("No discards left this round!")
            invalid_action = True
        else:
            logging.info(f"Action: {action_arg} cards at indices {target_card_indices}")
            try:
                self.client.send_message(
                    "play_hand_or_discard",
                    {
                        "action": action_arg,
                        "cards": target_card_indices,
                    },
                )
            except balatrobot.exceptions.InvalidCardIndexError:
                logging.error(f"Invalid card indices {target_card_indices} selected!")
                invalid_action = True
            except balatrobot.exceptions.InvalidActionError:
                logging.error(
                    f"Invalid action attempted: {action_arg} with cards {target_card_indices}!"
                )
                invalid_action = True

        # Check game state after action
        full_obs = self.client.send_message("get_game_state", {})
        observation = self._get_obs()
        state_id = observation["state"]
        game_state = balatrobot.enums.State(state_id).name

        terminated = False
        prev_chips = self.current_chips
        self.current_chips = full_obs["game"]["chips"]
        chips_gain = self.current_chips - prev_chips
        reward = chips_gain / chips_target

        if game_state == "GAME_OVER":
            reward -= 1
            terminated = True
            self.log(f"Lost the blind {reward}")
        elif game_state == "ROUND_EVAL":
            # won the blind
            reward += 1
            terminated = True
            self.log(f"Won the blind {reward}")
        elif invalid_action:
            reward = -0.5
            self.log(f"Invalid action {reward}")
        else:
            self.log(f"Valid action {reward}")

        return observation, reward, terminated, False, {}

def register_mini_env():
    gym.register(
        id="Balatro-Blind-Mini-v0",
        entry_point="mini_env:BalatroEnv",
        max_episode_steps=1_000,
    )

if __name__ == "__main__":
    env = gym.make("Balatro-Blind-Mini-v0")
    obs, info = env.reset()
    _output_key_diff(env, obs)
    check_env(env)
