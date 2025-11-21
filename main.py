import logging
from balatrobot.client import BalatroClient
from balatrobot.exceptions import BalatroError
from typing import Optional
import numpy as np
import gymnasium as gym
import contextlib
from enum import Enum, auto, EnumMeta
import pathlib
from gymnasium.spaces import (
    OneOf,
    Dict,
    Tuple,
    Discrete,
    MultiBinary,
    Box,
    MultiDiscrete,
)
from gymnasium.utils.env_checker import check_env
import balatrobot.enums
import json
from contextlib import nullcontext

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _create_numbered_enum(class_name, action_keys):
    members = [(key.upper(), i) for i, key in enumerate(action_keys)]
    return Enum(class_name, members)

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
BLIND_STATUSES = ["Current", "Upcoming", "Defeated", "Select"]

N_JOKERS = len(balatrobot.enums.Jokers)
N_VOUCHERS = len(balatrobot.enums.Vouchers)
N_DECKS = len(balatrobot.enums.Decks)

GAME_ACTION_SPACE = {
    "skip_or_select_blind": Discrete(2),
    "play_hand_or_discard": Dict(
        {
            "action": Discrete(2),  # 0: Play Hand, 1: Discard
            "cards": MultiBinary(
                MAX_PLAY_HAND_SIZE
            ),  # Binary mask for cards to play or
        }
    ),
    "shop_next_round": Discrete(1),
    "shop_buy_card": Discrete(4),
    "shop_buy_and_use_card": Discrete(MAX_SHOP_JOKERS),
    "shop_reroll": Discrete(1),
    "shop_redeem_voucher": Discrete(1),
    "sell_joker": Discrete(UNBOUNDED_MAX),
    "sell_consumable": Discrete(UNBOUNDED_MAX),
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

CARD_RANKS = [
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "Jack",
    "Queen",
    "King",
    "Ace",
]

TAGS = [
    "Uncommon Tag",
    "Rare Tag",
    "Negative Tag",
    "Foil Tag",
    "Holographic Tag",
    "Polychrome Tag",
    "Investment Tag",
    "Voucher Tag",
    "Boss Tag",
    "Standard Tag",
    "Charm Tag",
    "Meteor Tag",
    "Buffoon Tag",
    "Handy Tag",
    "Garbage Tag",
    "Ethereal Tag",
    "Coupon Tag",
    "Double Tag",
    "Juggle Tag",
    "D6 Tag",
    "Top-up Tag",
    "Speed Tag",
    "Orbital Tag",
    "Economy Tag",
]

CARD_SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]

N_CONSUMABLES = len(balatrobot.enums.Consumables)
N_STAKES = len(balatrobot.enums.Stakes)


CARD_MODIFIERS = ["Enancement", "Edition", "Seal"]

CARD_FEATURE_COUNTS = [
    len(CARD_RANKS),
    len(CARD_SUITS),
    len(balatrobot.enums.Enhancements),
    len(balatrobot.enums.Editions),
    len(balatrobot.enums.Seals),
]

HAND_NVEC = np.tile(CARD_FEATURE_COUNTS, (MAX_HELD_HAND_SIZE, 1))

HIGHLIGHTED_CARDS_NVEC = np.tile(CARD_FEATURE_COUNTS, (MAX_PLAY_HAND_SIZE, 1))


class ReverseMap(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We calculate the inverse once upon creation
        # Note: This assumes the map is static (won't change after build_map returns)
        self._inverse = {v: k for k, v in self.items()}

    @property
    def inverse(self):
        return self._inverse


def _build_map(items: list[str], start_index: int = 1) -> dict:
    if isinstance(items, list):
        item_names = items
    elif isinstance(items, EnumMeta):
        item_names = [item.name for item in items]
    else:
        raise ValueError("items must be a list or an EnumMeta")

    name_map = {name: i for i, name in enumerate(item_names, start=start_index)}

    return ReverseMap(name_map)


def _process_card_features(card):
    card_rank = card["base"]["id"]
    card_rank_id = card_rank - 2

    card_suit = card["base"]["suit"]
    card_suit_id = BalatroEnv.CARD_SUITS[card_suit]

    card_enhancement = card["ability"]["name"].split(" ")[0]
    card_enhancement_id = BalatroEnv.ENHANCEMENTS[card_enhancement]

    card_edition = card["edition"]["key"]  # this is unverified
    card_edition_id = BalatroEnv.EDITIONS[card_edition]

    card_seal = card["seal"]
    card_seal_id = BalatroEnv.SEALS[card_seal]

    return [
        card_rank_id,
        card_suit_id,
        card_enhancement_id,
        card_edition_id,
        card_seal_id,
    ]


class BalatroEnv(gym.Env):
    ACTIONS = _build_map(GAME_ACTION_NAMES, start_index=0)
    JOKERS = [joker.name for joker in balatrobot.enums.Jokers]
    JOKER_MAP = _build_map(JOKERS)

    CONSUMABLES = [consumable.name for consumable in balatrobot.enums.Consumables]
    CONSUMABLES_MAP = _build_map(CONSUMABLES)

    PACKS_MAP = _build_map(PACKS)

    JOKERS_AND_CONSUMABLES_MAP = _build_map(JOKERS + CONSUMABLES)
    STATES = balatrobot.enums.State

    DECKS = [deck.value for deck in balatrobot.enums.Decks]
    DECKS_MAP = _build_map(DECKS)

    CARD_SUITS = _build_map(CARD_SUITS, start_index=0)
    CARD_MODIFIERS = _build_map(CARD_MODIFIERS)

    ENHANCEMENTS = _build_map(balatrobot.enums.Enhancements, start_index=0)
    EDITIONS = _build_map(balatrobot.enums.Editions, start_index=0)
    SEALS = _build_map(balatrobot.enums.Seals, start_index=0)

    VOUCHERS = _build_map(balatrobot.enums.Vouchers, start_index=1)

    BLIND_STATUSES = _build_map(BLIND_STATUSES, start_index=0)
    BOSS_BLINDS_MAP = _build_map(BOSS_BLINDS, start_index=0)

    TAGS_MAP = _build_map(TAGS)

    STATE_ACTIONS = {
        "MENU": ["start_run"],
        "BLIND_SELECT": [
            "skip_or_select_blind",
            "sell_joker",
            "sell_consumable",
            "use_consumable",
        ],
        "SELECTING_HAND": [
            "play_hand_or_discard",
            "sell_joker",
            "sell_consumable",
            "use_consumable",
        ],
        "ROUND_EVAL": ["cash_out", "sell_joker", "sell_consumable", "use_consumable"],
        "SHOP": [
            "shop_next_round",
            "shop_buy_card",
            "shop_buy_and_use_card",
            "shop_reroll",
            "shop_redeem_voucher",
            "sell_joker",
        ],
        "GAME_OVER": ["go_to_menu"],
    }

    def __init__(self, deck: str = "Red Deck", stake: int = 1):
        self.deck = deck
        self.stake = stake
        self.client = BalatroClient()
        self.client.connect()

        # Order taken from balatrobot.src.balatrobot.enums.py.Actions
        self.action_space = Dict(
            {
                "action_type": Discrete(len(self.ACTIONS)),
                "action args": Dict(GAME_ACTION_SPACE),
            }
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
                "state": Discrete(len(balatrobot.enums.State)),
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
                            }
                        ),
                        "dollars": Box(
                            low=-1_000, high=1_000, shape=(1,), dtype=np.int32
                        ),
                        "hands_played": Box(
                            low=0, high=1_000, shape=(1,), dtype=np.int32
                        ),
                        "log_chips": Box(low=0, high=100, shape=(1,), dtype=np.float32),
                        "round": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                        "deck": Discrete(N_DECKS),
                        "stake": Box(low=1, high=N_STAKES, shape=(1,), dtype=np.int32),
                        "win_ante": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                        "interest_cap": Box(
                            low=0, high=100, shape=(1,), dtype=np.int32
                        ),
                        "skips": Box(low=0, high=100, shape=(1,), dtype=np.int32),
                        "base_reroll_cost": Box(
                            low=0, high=100, shape=(1,), dtype=np.int32
                        ),
                        "interest_amount": Box(
                            low=0, high=100, shape=(1,), dtype=np.int32
                        ),
                    }
                ),
                # TODO: ADD HAND LEVELS
                "hand_cards": MultiDiscrete(nvec=HAND_NVEC, dtype=np.int32),
                "highlighted_cards": MultiDiscrete(
                    nvec=HIGHLIGHTED_CARDS_NVEC, dtype=np.int32
                ),
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
                                "status": Discrete(len(BLIND_STATUSES)),
                            }
                        ),
                        "big": Dict(
                            {
                                "tag_name": Discrete(len(balatrobot.enums.Tags)),
                                "log_score": Box(
                                    low=0, high=100, shape=(1,), dtype=np.float32
                                ),
                                "status": Discrete(len(BLIND_STATUSES)),
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
        if getattr(self, "client", None):
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
                logger.error(f"Unexpected balatro function error: {e}")
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
        current_state = balatrobot.enums.State(game_obs["state"])
        logging.info(f"Current game state: {current_state.name}")
        joker_names = [
            joker["config"]["center_key"] for joker in game_obs["jokers"]["cards"]
        ]
        state = game_obs["state"]
        game = game_obs["game"]
        blinds = game_obs["blinds"]

        observation = {}

        # jokers_count
        observation["jokers_count"] = np.array(
            [game_obs["jokers"]["config"]["card_count"]], dtype=np.int32
        )

        # jokers_limit
        observation["jokers_limit"] = np.array(
            [game_obs["jokers"]["config"]["card_limit"]], dtype=np.int32
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
        if current_state == "SHOP":
            shop_booster = game_obs["shop_booster"]
            for i, booster in enumerate(shop_booster["cards"]):
                booster_name = booster["label"]
                obs_shop_booster[i] = PacksEnum[booster_name].value
        observation["shop_booster"] = obs_shop_booster

        # consumables
        obs_consumables = np.zeros(UNBOUNDED_MAX, dtype=np.int32)
        for i, consumable in enumerate(game_obs["consumables"]["cards"]):
            consumable_name = consumable["config"]["center_key"]
            obs_consumables[i] = self.CONSUMABLES_MAP[consumable_name].value
        observation["consumables"] = obs_consumables

        # shop_jokers
        obs_shop_jokers = np.zeros(MAX_SHOP_JOKERS, dtype=np.int32)
        if current_state == "SHOP":
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
        obs_current_round = {
            key: np.array([current_round[key]], dtype=np.int32)
            for key in current_round_keys
        }
        game = game_obs["game"]
        game_keys = [
            "hands_played",
            "round",
            "dollars",
            "stake",
            "win_ante",
            "interest_cap",
            "skips",
            "base_reroll_cost",
            "interest_amount",
        ]
        obs_game = {key: np.array([game[key]], dtype=np.int32) for key in game_keys}
        obs_game["current_round"] = obs_current_round
        chips = game_obs["game"]["chips"]
        obs_game["log_chips"] = np.array([np.log(chips + 1)], dtype=np.float32)
        obs_game["deck"] = self.DECKS_MAP[game_obs["game"]["selected_back"]["name"]]

        observation["game"] = obs_game

        # hand_cards
        obs_hand_cards = np.zeros(
            (MAX_HELD_HAND_SIZE, len(CARD_FEATURE_COUNTS)), dtype=np.int32
        )
        if current_state == "SELECTING_HAND":
            hand = game_obs["hand"]
            for i, card in enumerate(hand["cards"]):
                features = _process_card_features(card)
                obs_hand_cards[i] = features
        observation["hand_cards"] = obs_hand_cards

        # highlighted_cards
        highlighted_cards_obs = np.zeros(
            (MAX_PLAY_HAND_SIZE, len(CARD_FEATURE_COUNTS)), dtype=np.int32
        )
        if current_state == "SELECTING_HAND":
            pass
            highlighted_cards = [card for card in hand["cards"] if card["highlighted"]]
            for i, card in enumerate(highlighted_cards):
                features = _process_card_features(card)
                highlighted_cards_obs[i] = features
        observation["highlighted_cards"] = highlighted_cards_obs

        # shop_vouchers
        voucher_id = 0
        if current_state == "SHOP":
            voucher_name = game_obs["shop_vouchers"]["cards"][0]["config"]["center_key"]
            voucher_id = self.VOUCHERS[voucher_name]
        observation["shop_vouchers"] = voucher_id

        # blinds
        blinds = game_obs["blinds"]
        obs_blind = {}
        for blind_type in ["boss", "small", "big"]:
            blind_features = {}
            blind = blinds[blind_type]
            blind_features["status"] = self.BLIND_STATUSES[blind["status"]]
            blind_features["log_score"] = np.array(
                [np.log(blind["score"] + 1)], dtype=np.float32
            )

            if blind_type == "boss":
                blind_features["name"] = self.BOSS_BLINDS_MAP[blind["name"]]
            else:
                blind_features["tag_name"] = self.TAGS_MAP[blind["tag_name"]]

            obs_blind[blind_type] = blind_features

        observation["blinds"] = obs_blind

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
        action_id = action["action_type"]
        action_args = action["action args"]
        action_name = self.ACTIONS.inverse[int(action_id)]
        args = action_args[action_name]

        observation = self._get_obs()
        state_id = observation["state"]
        state_name = self.STATES(state_id).name

        if action_name not in self.STATE_ACTIONS[state_name]:
            logging.error(f"Action {action_name} not valid in state {state_name}!")
            return observation, -1, False, False, self._get_info()
        else:
            logging.info(f"Executing action: {action_name} from state {state_name}.")
            # Invalid action for current state TODO: Early stop here

        print(f"Taking action: {action_name} with args: {action_args}")

        if action_name == "SKIP_OR_SELECT_BLIND":
            skip_or_select = action_args["SKIP_OR_SELECT_BLIND"]
            action_arg = "skip" if skip_or_select == 0 else "select"
            self.client.send_message("skip_or_select_blind", {"action": action_arg})
        elif action_name == "PLAY_HAND_OR_DISCARD":
            play_or_discard = action_args["PLAY_HAND_OR_DISCARD"]["action"]
            action_arg = "play_hand" if play_or_discard == 0 else "discard"
            card_mask = action_args["PLAY_HAND_OR_DISCARD"]["cards"]
            selected_cards = [
                i for i, selected in enumerate(card_mask) if selected == 1
            ]
            self.client.send_message(
                "play_hand_or_discard",
                {
                    "action": action_arg,
                    "cards": selected_cards,
                },
            )
        elif action_name == "SHOP_NEXT_ROUND":
            self.client.send_message("next_round", {})
        elif action_name == "SHOP_BUY_CARD":
            self.client.send_message("buy_card", {"index": args})
        elif action_name == "SHOP_BUY_AND_USE_CARD":
            self.client.send_message("buy_and_use_card", {"index": args})
        elif action_name == "SHOP_REROLL":
            self.client.send_message("reroll", {})
        elif action_name == "SHOP_REDEEM_VOUCHER":
            self.client.send_message("redeem_voucher", {"index": args})
        elif action_name == "SELL_JOKER":
            self.client.send_message("sell_joker", {"index": args})
        elif action_name == "SELL_CONSUMABLE":
            self.client.send_message("sell_consumable", {"index": args})
        elif action_name == "USE_CONSUMABLE":
            self.client.send_message("use_consumable", {"index": args})
        else:
            logging.error(f"Unknown action {action_name} in state {state_name}.")

        reward = 0
        terminated = False
        if state_name == "GAME_OVER":
            logging.info("GAME OVER REACHED!!!!!!!!!!")
            reward = 1
            terminated = True

        return self._get_obs(), reward, terminated, False, self._get_info()


def register_env():
    gym.register(id="Balatro-v0", entry_point="main:BalatroEnv", max_episode_steps=10)


gym.register(id="Balatro-v0", entry_point="main:BalatroEnv", max_episode_steps=10)

def test_env():
    env = BalatroEnv()
    try:
        obs, info = env.reset()
        obs = env._get_obs()
    finally:
        env.close()

    # print(obs)


def save_data(file_name: str = "response", client: Optional[BalatroClient] = None):
    balatro_client = nullcontext(client) if client else BalatroClient()

    with balatro_client as client:
        try:
            game_state = client.send_message("get_game_state", {})
            # print(game_state)

            output_files = pathlib.Path(".").glob(f"output/{file_name}*.json")
            num_existing_outputs = len(list(output_files))
            output_json = f"output/{file_name}_{num_existing_outputs}.json"
            with open(output_json, "w") as f:
                json.dump(game_state, f, indent=4)
            logging.info(f"Saved game state to {output_json}")

        except Exception as e:
            logger.error(f"Unexpected save_data error: {e}")


def _output_key_diff(env, obs):
    for key, space in env.observation_space.items():
        if key not in obs:
            print(f"❌ MISSING KEY: {key}")
            continue

        val = obs[key]

        # check if this specific item is valid
        is_valid = space.contains(val)

        if not is_valid:
            print(f"❌ INVALID: {key}")
            print(f"   Expected: {space}")
            print(f"   Got Type: {type(val)}")
            print(f"   Got Val : {val}")

            # Common check for shape/dtype issues
            if isinstance(val, (np.ndarray, np.generic)):
                print(
                    f"   Got Shape: {val.shape if hasattr(val, 'shape') else 'Scalar'}"
                )
                print(f"   Got Dtype: {val.dtype}")
        else:
            print(f"✅ OK: {key}")


def test_environment():
    env = gym.make("Balatro-v0")

    try:
        obs, info = env.reset()
        _output_key_diff(env, obs)
        check_env(env.unwrapped)
    except Exception as e:
        raise e
    finally:
        save_data(file_name="test_fail_output", client=env.unwrapped.client)
        env.close()

    # print(obs)


if __name__ == "__main__":
    # save_data()
    register_env()
    test_environment()
