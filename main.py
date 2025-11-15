import logging
from balatrobot.client import BalatroClient
from balatrobot.exceptions import BalatroError
from typing import Optional
import numpy as np
import gymnasium as gym
import contextlib
from enum import Enum, auto, EnumMeta
from gymnasium.spaces import (
    OneOf,
    Dict,
    Tuple,
    Discrete,
    MultiBinary,
    Box,
    MultiDiscrete,
)
import balatrobot.enums
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _create_numbered_enum(class_name, action_keys):
    members = [(key.upper(), i) for i, key in enumerate(action_keys)]
    return Enum(class_name, members)


# Actions = _create_numbered_emum(
#     [
#         "GET_GAME_STATE",
#         "GO_TO_MENU",
#         "START_RUN",
#         "SKIP_OR_SELECT_BLIND",
#         "PLAY_HAND_OR_DISCARD",
#         "REARRANGE_HAND",
#         "REARRANGE_CONSUMABLES",
#         "CASH_OUT",
#         "SHOP",
#         "NEXT_ROUND",
#         "BUY_CARD",
#         "BUY_AND_USE_CARD",
#         "REROLL",
#         "SELL_JOKER",
#         "SELL_CONSUMABLE",
#         "USE_CONSUMABLE",
#     ]
# )

# Decks = _create_numbered_emum(["RED_DECK", "BLUE_DECK", "YELLOW_DECK"])


# class Decks(Enum):
#     RED_DECK = 0
#     BLUE_DECK = 1


"""
# Define two disjoint action spaces: one for when in the shop, another for when playing a blind
shop_action_space = Tuple([
    # Subspace 1
    Box(low=0.0, high=10.0),
    # Subspace 2
    Discrete(3)
])

blind_action_space = Tuple([
    # Subspace 3
    Box(low=-5.0, high=5.0),
    # Subspace 4
    Discrete(6)
])

# Use OneOf to define the overall action space
overall_action_space = OneOf([shop_action_space, blind_action_space])

"""


# class Actions(Enum):
#     # Source: https://coder.github.io/balatrobot/protocol-api/#game-states
#     SKIP_OR_SELECT_BLIND = 1
#     PLAY_HAND_OR_DISCARD = 2
#     SHOP_NEXT_ROUND = 3
#     SELL_JOKER = 4
#     SELL_CONSUMABLE = 5
#     USE_CONSUMABLE = 6

MAX_HELD_HAND_SIZE = 20
MAX_SHOP_JOKERS = 4
UNBOUNDED_MAX = 100
MAX_PLAY_HAND_SIZE = 5
BOSS_BLINDS = [
    "The Hook",
    "The Ox",
    "The House",
    "The Wall",
    "The Wheel",
    "The Arm",
    "The Club",
    "The Fish",
    "The Psychic",
    "The Goad",
    "The Water",
    "The Window",
    "The Manacle",
    "The Eye",
    "The Mouth",
    "The Plant",
    "The Serpent",
    "The Pillar",
    "The Needle",
    "The Head",
    "The Tooth",
    "The Flint",
    "The Mark",
    "Amber Acorn",
    "Verdant Leaf",
    "Violet Vessel",
    "Crimson Heart",
    "Cerulean Bell",
]
N_BOSS_BLINDS = len(BOSS_BLINDS)
BLIND_STATUSES = ["Current", "Upcoming", "Defeated"]

N_JOKERS = len(balatrobot.enums.Jokers)
N_VOUCHERS = len(balatrobot.enums.Vouchers)
N_DECKS = len(balatrobot.enums.Decks)

GAME_ACTION_SPACE = {
    "SKIP_OR_SELECT_BLIND": Discrete(2),
    "PLAY_HAND_OR_DISCARD": Dict(
        {
            "action": Discrete(2),  # 0: Play Hand, 1: Discard
            "cards": MultiBinary(
                MAX_PLAY_HAND_SIZE
            ),  # Binary mask for cards to play or
        }
    ),
    "SHOP_NEXT_ROUND": Dict({}),
    "SHOP_BUY_CARD": Discrete(MAX_SHOP_JOKERS),
    "SHOP_BUY_AND_USE_CARD": Discrete(MAX_SHOP_JOKERS),
    "SHOP_REROLL": Dict({}),
    "SHOP_REDEEM_VOUCHER": Dict(),
    "SELL_JOKER": Discrete(UNBOUNDED_MAX),
    "SELL_CONSUMABLE": Discrete(UNBOUNDED_MAX),
}

GAME_ACTION_NAMES = list(GAME_ACTION_SPACE.keys())
ActionsEnum = _create_numbered_enum("ActionsEnum", GAME_ACTION_NAMES)

PACKS = [
    "Buffoon Pack",
    "Mega Buffoon Pack",
    "Jumbo Buffoon Pack",
    "Arcana Pack",
    "Mega Arcana Pack",
    "Jumbo Arcana Pack",
    "Celestial Pack",
    "Mega Celestial Pack",
    "Jumbo Celestial Pack",
    "Standard Pack",
    "Mega Standard Pack",
    "Jumbo Standard Pack",
    "Spectral Pack",
    "Mega Spectral Pack",
    "Jumbo Spectral Pack",
]
PacksEnum = _create_numbered_enum("PacksEnum", PACKS)

PLANET_CARDS = [
    "Pluto",
    "Mercury",
    "Uranus",
    "Venus",
    "Saturn",
    "Jupiter",
    "Earth",
    "Mars",
    "Neptune",
    "Planet X",
    "Ceres",
    "Eris",
]

TAROT_CARDS = [
    "The Fool",
    "The Magician",
    "The High Priestess",
    "The Empress",
    "The Emperor",
    "The Hierophant",
    "The Lovers",
    "The Chariot",
    "Justice",
    "The Hermit",
    "The Wheel of Fortune",
    "Strength",
    "The Hanged Man",
    "Death",
    "Temperance",
    "The Devil",
    "The Tower",
    "The Star",
    "The Moon",
    "The Sun",
    "Judgement",
    "The World",
]

# CONSUMABLES = PLANET_CARDS + TAROT_CARDS

RANKS = [
    "Ace",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
]

SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]

N_CONSUMABLES = len(balatrobot.enums.Consumables)
N_STAKES = len(balatrobot.enums.Stakes)


CARD_MODIFIERS = ["Enancement", "Edition", "Seal"]

CARD_FEATURE_COUNTS = [
    len(RANKS),
    len(SUITS),
    len(balatrobot.enums.Enhancements),
    len(balatrobot.enums.Editions),
    len(balatrobot.enums.Seals),
]

HAND_NVEC = np.tile(CARD_FEATURE_COUNTS, (MAX_HELD_HAND_SIZE, 1))

HIGHLIGHTED_CARDS_NVEC = np.tile(CARD_FEATURE_COUNTS, (MAX_PLAY_HAND_SIZE, 1))


def _build_map(items) -> dict:
    if isinstance(items, list):
        item_names = items
    elif isinstance(items, EnumMeta):
        item_names = [item.name for item in items]
    else:
        raise ValueError("items must be a list or an EnumMeta")

    name_map = {name: i for i, name in enumerate(item_names, start=1)}

    return name_map


class BalatroEnv(gym.Env):
    JOKERS = [joker.name for joker in balatrobot.enums.Jokers]
    JOKER_MAP = _build_map(JOKERS)

    CONSUMABLES = [consumable.name for consumable in balatrobot.enums.Consumables]
    CONSUMABLES_MAP = _build_map(CONSUMABLES)

    PACKS_MAP = _build_map(PACKS)

    JOKERS_AND_CONSUMABLES_MAP = _build_map(JOKERS + CONSUMABLES)
    STATES = balatrobot.enums.State

    DECKS = [deck.value for deck in balatrobot.enums.Decks]
    DECKS_MAP = _build_map(DECKS)

    def __init__(self, deck: str = "Red Deck", stake: int = 1):
        self.deck = deck
        self.stake = stake
        self.client = BalatroClient()
        self.client.connect()

        # Order taken from balatrobot.src.balatrobot.enums.py.Actions
        self.action_space = Dict(
            {"action_type": Discrete(len(self.ACTIONS))} + GAME_ACTION_NAMES
        )

        self.observation_space = Dict(
            {
                "jokers_count": Box(
                    low=0, high=UNBOUNDED_MAX, shape=(1,), dtype=np.int32
                ),
                "jokers_limit": Box(
                    low=0, high=UNBOUNDED_MAX, shape=(1,), dtype=np.int32
                ),
                "jokers": Box(
                    low=0, high=len(self.JOKERS), shape=(UNBOUNDED_MAX,), dtype=np.int32
                ),
                # "hand": Box(low=0, high=52, shape=(MAX_HAND_SIZE,), dtype=np.int32),
                "shop_booster": Box(low=0, high=len(PACKS), shape=(2,), dtype=np.int32),
                "consumables": Box(
                    low=0,
                    high=len(self.CONSUMABLES_MAP),
                    shape=(UNBOUNDED_MAX,),
                    dtype=np.int32,
                ),
                "shop_jokers": Box(
                    low=0, high=N_JOKERS, shape=(MAX_SHOP_JOKERS,), dtype=np.int32
                ),
                "state": Discrete(len(balatrobot.enums.GameStates)),
                "game": Dict(
                    {
                        "current_round": Dict(
                            {
                                "hands_left": Box(
                                    low=1, high=100, shape=(1,), dtype=np.int32
                                ),
                                "reroll_cost": Box(
                                    low=0, high=100, shape=(1,), dtype=np.int32
                                ),
                                "free_rerolls": Box(
                                    low=0, high=10, shape=(1,), dtype=np.int32
                                ),
                                "discards_left": Box(
                                    low=0, high=100, shape=(1,), dtype=np.int32
                                ),
                                "hands_played": Box(
                                    low=0, high=100, shape=(1,), dtype=np.int32
                                ),
                                "discards_used": Box(
                                    low=0, high=100, shape=(1,), dtype=np.int32
                                ),
                                "dollars": Box(
                                    low=-1_000, high=1_000, shape=(1,), dtype=np.int32
                                ),
                            }
                        ),
                        "hands_played": Box(
                            low=0, high=1_000, shape=(1,), dtype=np.int32
                        ),
                        "log_chips": Box(low=0, high=100, shape=(1,), dtype=np.float32),
                        "rounds": Box(low=1, high=100, shape=(1,), dtype=np.int32),
                        "deck": Discrete(N_DECKS),
                    }
                ),
                # TODO: ADD HAND LEVELS
                "hand_cards": MultiDiscrete(nvec=HAND_NVEC, dtype=np.int32),
                "highlighted_cards": MultiDiscrete(
                    nvec=HIGHLIGHTED_CARDS_NVEC, dtype=np.int32
                ),
                "stake": Box(low=1, high=N_STAKES, shape=(1,), dtype=np.int32),
                "win_ante": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "interest_cap": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "skips": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "base_reroll_cost": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "interest_amount": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "round": Box(low=1, high=100, shape=(1,), dtype=np.int32),
                "shop_vouchers": Discrete(N_VOUCHERS + 1),
                "blinds": Dict(
                    {
                        "boss": Dict(
                            {
                                "name": Discrete(N_BOSS_BLINDS),
                                "log_score": Box(
                                    low=0, high=100, shape=(1,), dtype=np.float32
                                ),
                                "status": Discrete(len(BLIND_STATUSES)),
                            }
                        ),
                        "small": Dict(
                            {
                                "tag_name": Discrete(len(balatrobot.enums.Tags)),
                                "log_score": Box(
                                    low=0, high=100, shape=(1,), dtype=np.float32
                                ),
                            }
                        ),
                        "large": Dict(
                            {
                                "tag_name": Discrete(len(balatrobot.enums.Tags)),
                                "log_score": Box(
                                    low=0, high=100, shape=(1,), dtype=np.float32
                                ),
                            }
                        ),
                    }
                ),
            }
        )

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
    def _start_game(self, seed: str = "0000155"):
        self.seed = seed
        self.client.send_message("go_to_menu", {})
        self.client.send_message(
            "start_run",
            {
                "deck": self.deck,
                "stake": self.stake,
                "seed": seed,
            },
        )

    ### Environment functions
    @balatro_function
    def _get_obs(self):
        game_obs = self.client.send_message("get_game_state", {})
        joker_names = [
            joker["config"]["center_key"] for joker in game_obs["jokers"]["cards"]
        ]
        hand = game_obs["hand"]
        shop_booster = game_obs["shop_booster"]
        state = game_obs["state"]
        game = game_obs["game"]
        shop_vouchers = game_obs["shop_vouchers"]
        blinds = game["blinds"]

        observation = {}

        # jokers_count
        observation["jokers_count"] = np.array(
            game_obs["jokers"]["config"]["card_count"], dtype=np.int32
        )

        # jokers_limit
        observation["jokers_limit"] = np.array(
            game_obs["jokers"]["config"]["card_limit"], dtype=np.int32
        )

        # jokers
        obs_jokers = np.zeros(UNBOUNDED_MAX, dtype=np.int32)
        for i, joker_name in enumerate(joker_names):
            if joker_name not in self.JOKERS:
                raise ValueError(f"Unknown joker: {joker_name}")
            obs_jokers[i] = self.JOKER_MAP[joker_name]
        observation["jokers"] = obs_jokers

        # shop_booster
        obs_shop_booster = np.zeros(2, dtype=np.int32)
        for i, booster in enumerate(shop_booster["cards"]):
            booster_name = booster["label"]
            obs_shop_booster[i] = PacksEnum[booster_name].value
        observation["shop_booster"] = obs_shop_booster

        # consumables
        obs_consumables = np.zeros(UNBOUNDED_MAX, dtype=np.int32)
        for i, consumable in enumerate(game_obs["consumables"]["cards"]):
            consumable_name = consumable["config"]["center_key"]
            obs_consumables[i] = self.CONSUMABLES_MAP[consumable_name].value

        # shop_jokers
        obs_shop_jokers = np.zeros(MAX_SHOP_JOKERS, dtype=np.int32)
        for i, card in enumerate(game_obs["shop_jokers"]["cards"]):
            card_name = card["config"]["center_key"]
            obs_shop_jokers[i] = self.JOKERS_AND_CONSUMABLES_MAP[card_name].value
        observation["shop_jokers"] = obs_shop_jokers

        # state
        observation["state"] = state

        # game
        current_round = game_obs["game"]["current_round"]
        current_round_keys = [
            "hands_left",
            "reroll_cost",
            "free_rerolls",
            "discards_left",
            "hands_played",
            "discards_used",
        ]
        obs_current_round = {key: current_round[key] for key in current_round_keys}
        game = game_obs["game"]
        game_keys = ["hands_played", "log_chips", "rounds", "decks", "dollars"]
        obs_game = {key: game[key] for key in game_keys}
        obs_game["current_round"] = obs_current_round
        chips = game_obs["game"]["chips"]
        obs_game["log_chips"] = np.log(chips + 1).astype(np.float32)
        observation["game"] = obs_game
        obs_game["deck"] = self.DECKS_MAP(game_obs["game"]["selected_back"]["name"])
        observation["game"] = obs_game

        # hand_cards
        obs_hand_cards = np.zeros(
            (MAX_HELD_HAND_SIZE, len(CARD_FEATURE_COUNTS)), dtype=np.int32
        )

        for i, card in enumerate(hand["cards"]):
            # card_rank =
            pass

        return observation

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


def test_env():
    env = BalatroEnv()
    try:
        obs, info = env.reset()
        obs = env._get_obs()
    finally:
        env.client.disconnect()

    # print(obs)


def test_client():
    with BalatroClient() as client:
        try:
            game_state = client.send_message("get_game_state", {})
            print(game_state)

            with open("shop_response.json", "w") as f:
                json.dump(game_state, f, indent=4)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    test_client()
