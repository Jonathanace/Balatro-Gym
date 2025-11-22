import gymnasium as gym


class CombinatorialActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_held_cards=8):
        super().__init__(env)
        self.max_held_cards = max_held_cards

        # LOGIC:
        # We have 'max_held_cards' (N).
        # There are 2^N possible combinations of cards to select.
        # We can perform 2 types of moves with those cards: PLAY or DISCARD.
        # Total Actions = 2 * (2^N) = 2^(N+1)

        # Example for 8 cards:
        # 2^8 = 256 card combos.
        # 256 * 2 = 512 total unique buttons.
        self.total_actions = 2 ** (self.max_held_cards + 1)

        # The Agent sees one simple list of 512 buttons
        self.action_space = gym.spaces.Discrete(self.total_actions)

    def action(self, action_int):
        # 1. Determine if this is a PLAY or DISCARD move
        # The split happens exactly in the middle.
        # 0 to 255 -> PLAY
        # 256 to 511 -> DISCARD
        threshold = 2**self.max_held_cards

        if action_int < threshold:
            play_val = 0  # Play
            card_mask_int = action_int
        else:
            play_val = 1  # Discard
            card_mask_int = action_int - threshold

        # 2. Convert the integer into a Binary List (Bitmask)
        # If card_mask_int is 5 (binary 101), we want [1, 0, 1, 0, 0...]
        target_cards = [(card_mask_int >> i) & 1 for i in range(self.max_held_cards)]

        # 3. Return the exact Dictionary your environment expects
        return {"play_or_discard": play_val, "target_cards": target_cards}
