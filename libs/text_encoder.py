from pathlib import Path
from typing import List

from janome.tokenizer import Tokenizer
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer


def get_tokenized_text_list(file_path, lang="en"):
    if lang == "en":
        tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    elif lang == "de":
        tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    else:
        raise NotImplementedError

    tokenized_text_list = []
    with file_path.open("r") as f:
        for line in f:
            line = line.strip()
            tokenized_text_list.append(tokenizer(line))

    return tokenized_text_list


class SpacedTextTokenizer:
    def __call__(self, text: str):
        return text.strip().split()


class JapaneseTokenizer:
    def __init__(self):
        self._tokenizer = Tokenizer()

    def __call__(self, text: str):
        return list(self._tokenizer.tokenize(text, wakati=True))


class TextEncoder:
    def __init__(
        self,
        tokenizer,
        vocab: Vocab,
    ):
        self._tokenizer = tokenizer
        self._vocab = vocab

    def get_bos_id(self):
        return self._vocab["<bos>"]

    def get_eos_id(self):
        return self._vocab["<eos>"]

    def encode(self, text: str):
        tokenized_text = self._tokenizer(text)
        return self._vocab.lookup_indices(tokenized_text)

    def decode(self, id_list: List[int], sep=""):
        word_list = self._vocab.lookup_tokens(id_list)
        return sep.join(word_list)


if __name__ == "__main__":
    from text_pair_dataset import get_vocab

    ja_word_freqs_path = Path("../multi30k_dataset/en_word_freqs.json").resolve()
    vocab = get_vocab(ja_word_freqs_path)
    text_encoder = TextEncoder(get_tokenizer("spacy", language="en_core_web_sm"), vocab)

    text = "A man lays on the bench to which a white dog is also tied ."
    print("text:", text)

    encoded_text = text_encoder.encode(text)
    print("encoded_text:", encoded_text)

    decoded_text = text_encoder.decode(encoded_text, sep=" ")
    print("decoded_text:", decoded_text)
