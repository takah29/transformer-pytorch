from pathlib import Path
from typing import List

from janome.tokenizer import Tokenizer
from torchtext.vocab import Vocab


def get_tokenized_text_list(file_path):
    tokenizer = EnglishTokenizer()
    tokenized_text_list = []
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            tokenized_text_list.append(tokenizer(line))

    return tokenized_text_list


class SepalatedTextTokenizer:
    def tokenize(self, text: str):
        return text.strip().split()


class EnglishTokenizer:
    def tokenize(self, text: str):
        if text[-1] == "." and text[-2] != " ":
            text = text[:-1] + " ."
        return text.split()


class JapaneseTokenizer:
    def __init__(self):
        self._tokenizer = Tokenizer()

    def tokenize(self, text: str):
        return list(self._tokenizer.tokenize(text, wakati=True))


class TextEncoder:
    def __init__(
        self,
        tokenizer,
        vocab: Vocab,
    ):
        self._tokenizer = tokenizer
        self._vocab = vocab

    def encode(self, text: str):
        tokenized_text = self._tokenizer.tokenize(text)
        return self._vocab.lookup_indices(tokenized_text)

    def decode(self, id_list: List[int], sep=""):
        word_list = self._vocab.lookup_tokens(id_list)
        return sep.join(word_list)


if __name__ == "__main__":
    from text_pair_dataset import get_vocab

    ja_word_freqs_path = Path("../word_freqs/dec_word_freqs.json").resolve()
    vocab = get_vocab(ja_word_freqs_path)
    text_encoder = TextEncoder(JapaneseTokenizer(), vocab)

    text = "お腹が痛いので遅れます。"
    print("text:", text)

    encoded_text = text_encoder.encode(text)
    print("encoded_text:", encoded_text)

    decoded_text = text_encoder.decode(encoded_text)
    print("decoded_text:", decoded_text)
