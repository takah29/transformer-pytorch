from pathlib import Path
import json
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.vocab import vocab


def get_vocab(word_freqs_file_path):
    with open(word_freqs_file_path, "r") as f:
        word_freqs_dict = json.load(f, object_pairs_hook=OrderedDict)

    voc = vocab(word_freqs_dict, specials=(["<pad>", "<unk>", "<bos>", "<eos>"]))
    voc.set_default_index(voc["<unk>"])

    return voc


class TextPairDataset(Dataset):
    def __init__(self, tokenized_text_list_1, tokenized_text_list_2, vocab_1, vocab_2, word_count):
        super().__init__()

        self._tokenized_text_list_1 = tokenized_text_list_1
        self._tokenized_text_list_2 = tokenized_text_list_2

        self._vocab_1 = vocab_1
        self._vocab_2 = vocab_2

        self._word_count = word_count

        self._text_transform_1 = TextPairDataset._get_transform(word_count, vocab_1)
        self._text_transform_2 = TextPairDataset._get_transform(word_count, vocab_2)

        # 2つのtextリストのうち要素数が少ない方のリストに合わせる。超過した要素は無視する
        self._n_data = min(len(self._tokenized_text_list_1), len(self._tokenized_text_list_2))

        self._pad_word_id_1 = vocab_1["<pad>"]
        self._pad_word_id_2 = vocab_2["<pad>"]

    def __getitem__(self, i):
        enc_input = self._tokenized_text_list_1[i]
        enc_input = self._text_transform_1([enc_input]).squeeze()

        target = self._tokenized_text_list_2[i]
        target = self._text_transform_2([target]).squeeze()

        dec_input = target[:-1]
        dec_target = target[1:]  # 右に1つずらす

        data = {
            "enc_input": {
                "text": enc_input,
                "mask": self._get_padding_mask(enc_input, self._pad_word_id_1),
            },
            "dec_input": {
                "text": dec_input,
                "mask": self._get_padding_mask(dec_input, self._pad_word_id_2),
            },
            "dec_target": dec_target,
        }

        return data

    def __len__(self):
        return self._n_data

    def get_vocab_size(self):
        return len(self._vocab_1), len(self._vocab_2)

    def get_pad_id(self):
        return self._pad_word_id_1, self._pad_word_id_2

    @staticmethod
    def _get_padding_mask(x, pad_id):
        return (x == pad_id).to(torch.bool)  # (token_size, )

    @staticmethod
    def _get_transform(word_count, vocab_data):
        text_transform = T.Sequential(
            T.VocabTransform(vocab_data),  # トークンに変換
            T.Truncate(word_count - 2),  # word_count - 2 を超過したデータを切り捨てる
            T.AddToken(token=vocab_data["<bos>"], begin=True),  # 先頭に'<bos>追加
            T.AddToken(token=vocab_data["<eos>"], begin=False),  # 終端に'<eos>'追加
            T.ToTensor(),  # テンソルに変換
            T.PadTransform(word_count, vocab_data["<pad>"]),  # word_countに満たない文章を'<pad>'で埋める
        )

        return text_transform

    @staticmethod
    def create(txt_file_path_1, txt_file_path_2, word_freqs_path_1, word_freqs_path_2):
        # 文章をトークン化してリスト化したデータを作成
        tokenized_text_list_1 = []
        with txt_file_path_1.open("r") as f:
            for line in f:
                tokenized_text_list_1.append(line.strip().split())
        tokenized_text_list_2 = []
        with txt_file_path_2.open("r") as f:
            for line in f:
                tokenized_text_list_2.append(line.strip().split())

        # 1文あたりの単語数
        word_count = 40

        vocab_1 = get_vocab(word_freqs_path_1)
        vocab_2 = get_vocab(word_freqs_path_2)

        return TextPairDataset(
            tokenized_text_list_1, tokenized_text_list_2, vocab_1, vocab_2, word_count
        )


if __name__ == "__main__":

    def test1():
        # 事前にbuild_small_parallel_enja_word_freqs.pyを実行してデータセットのダウンロードと単語辞書の作成を行っておく
        print("text_pair_dataset test")
        base_path = Path(__file__).resolve().parents[1] / "small_parallel_enja_dataset"
        en_txt_file_path = base_path / "en_train_texts.txt"
        ja_txt_file_path = base_path / "ja_train_texts.txt"
        en_word_freqs_path = base_path / "en_word_freqs.json"
        ja_word_freqs_path = base_path / "ja_word_freqs.json"

        text_pair_dataset = TextPairDataset.create(
            en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
        )
        print(*list(text_pair_dataset)[:10], sep="\n")

    test1()
