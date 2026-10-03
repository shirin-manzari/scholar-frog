class CharacterTokenizer:
    model_max_length = 512

    def encode(self, text, add_special_tokens=False, truncation=False):
        return list(text) + ([0, 0] if add_special_tokens else [])

    def num_special_tokens_to_add(self, pair=False):
        return 2
