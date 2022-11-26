from pathlib import Path
import json
from collections import OrderedDict
import pickle

import torch
from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.vocab import vocab


def get_tokenized_text_list(file_path):
    """small_parallel_enjaデータセットを単語リスト化したデータを返す"""
    tokenized_text_list = []
    with file_path.open("r") as f:
        for line in f:
            tokenized_text_list.append(line.strip().split())

    return tokenized_text_list


def get_vocab(word_freqs_file_path):
    with open(word_freqs_file_path, "r") as f:
        word_freqs_dict = json.load(f, object_pairs_hook=OrderedDict)

    voc = vocab(word_freqs_dict, specials=(["<pad>", "<unk>", "<bos>", "<eos>"]))
    voc.set_default_index(voc["<unk>"])

    return voc


class TextClassificationDataset(Dataset):
    def __init__(self, labeled_tokenized_text, vocab_data, word_count):
        super().__init__()

        self._tokenized_text_list = labeled_tokenized_text["tokenized_text_list"]
        self._class_label_list = labeled_tokenized_text["class_label_list"]
        self._class_name_list = labeled_tokenized_text["class_name_list"]
        self._vocab_data = vocab_data
        self._word_count = word_count

        self._text_transform = TextClassificationDataset._get_transform(word_count, vocab_data)
        self._n_data = len(self._tokenized_text_list)
        self._n_classes = len(self._class_name_list)
        self._pad_word_id = vocab_data["<pad>"]

    def __getitem__(self, i):
        enc_input = self._tokenized_text_list[i]
        enc_input = self._text_transform([enc_input]).squeeze()

        target = self._class_label_list[i]

        data = {
            "input": {
                "text": enc_input,
                "mask": self._get_padding_mask(enc_input, self._pad_word_id),
            },
            "target": target,
        }

        return data

    def __len__(self):
        return self._n_data

    def get_vocab_size(self):
        return len(self._vocab_data)

    def get_class_num(self):
        return self._n_classes

    @staticmethod
    def _get_padding_mask(x, pad_id):
        return (x != pad_id).to(torch.uint8)  # (token_size, )

    @staticmethod
    def _get_transform(word_count, vocab_data):
        text_transform = T.Sequential(
            T.VocabTransform(vocab_data),  # トークンに変換
            T.Truncate(word_count),  # word_countを超過したデータを切り捨てる
            T.AddToken(token=vocab_data["<bos>"], begin=True),  # 先頭に'<bos>追加
            T.AddToken(token=vocab_data["<eos>"], begin=False),  # 終端に'<eos>'追加
            T.ToTensor(),  # テンソルに変換
            T.PadTransform(word_count + 2, vocab_data["<pad>"]),  # word_countに満たない文章を'<pad>'で埋める
        )

        return text_transform

    @staticmethod
    def create(labeled_tokenized_text_pkl_path, word_freqs_path):
        # 文章をトークン化してリスト化したデータを作成
        with labeled_tokenized_text_pkl_path.open("rb") as f:
            labeled_tokenized_text = pickle.load(f)

        print(labeled_tokenized_text["class_name_list"])
        # 1文あたりの単語数
        word_count = 32

        vocab_data = get_vocab(word_freqs_path)

        return TextClassificationDataset(labeled_tokenized_text, vocab_data, word_count)


if __name__ == "__main__":

    def test1():
        # 事前にbuild_ldcc_word_freqs.pyを実行してデータセットのダウンロードと
        # データセットのpickleファイルと単語辞書の作成を行っておく
        print("text_classification_dataset test")
        base_path = Path(__file__).resolve().parents[1] / "ldcc_dataset"
        labeled_tokenized_text_pkl_path = base_path / "ldcc_tokenized_text_list.pkl"
        word_freqs_path = base_path / "ldcc_word_freqs.json"

        text_classification_dataset = TextClassificationDataset.create(
            labeled_tokenized_text_pkl_path, word_freqs_path
        )
        print(*list(text_classification_dataset)[:10], sep="\n")

    test1()
