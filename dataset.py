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
    def create(en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path):
        # 文章をトークン化してリスト化したデータを作成
        en_tokenized_text_list = get_tokenized_text_list(en_txt_file_path)
        ja_tokenized_text_list = get_tokenized_text_list(ja_txt_file_path)

        # 1文あたりの単語数
        word_count = 32

        en_vocab = get_vocab(en_word_freqs_path)
        ja_vocab = get_vocab(ja_word_freqs_path)

        return TextPairDataset(
            en_tokenized_text_list, ja_tokenized_text_list, en_vocab, ja_vocab, word_count
        )


class TextClassificationDataset(Dataset):
    def __init__(self, labeled_tokenized_text, vocab_data, word_count):
        super().__init__()

        self._tokenized_text_list = labeled_tokenized_text["tokenized_text_list"]
        self._class_num_list = labeled_tokenized_text["class_label_list"]
        self._class_name_list = labeled_tokenized_text["class_name_list"]
        self._vocab_data = vocab_data
        self._word_count = word_count

        self._text_transform = TextPairDataset._get_transform(word_count, vocab_data)
        self._n_data = len(self._tokenized_text_list)
        self._n_classes = len(self._class_num_list)
        self._pad_word_id = vocab_data["<pad>"]

    def __getitem__(self, i):
        enc_input = self._tokenized_text_list[i]
        enc_input = self._text_transform([enc_input]).squeeze()

        target = self._class_num_list[i]

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

        # 1文あたりの単語数
        word_count = 32

        vocab_data = get_vocab(word_freqs_path)

        return TextClassificationDataset(labeled_tokenized_text, vocab_data, word_count)


if __name__ == "__main__":
    def test1():
        # 事前にbuild_small_parallel_enja_word_freqs.pyを実行してデータセットのダウンロードと単語辞書の作成を行っておく
        print("text_pair_dataset test")
        en_txt_file_path = Path("dataset/small_parallel_enja-master/dev.en").resolve()
        ja_txt_file_path = Path("dataset/small_parallel_enja-master/dev.ja").resolve()
        en_word_freqs_path = Path("word_freqs_en.json").resolve()
        ja_word_freqs_path = Path("word_freqs_ja.json").resolve()

        text_pair_dataset = TextPairDataset.create(
            en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
        )
        print(*list(text_pair_dataset)[:10], sep="\n")
    test1()

    def test2():
        # 事前にbuild_ldcc_word_freqs.pyを実行してデータセットのダウンロードと
        # データセットのpickleファイルと単語辞書の作成を行っておく
        print("text_classification_dataset test")
        labeled_tokenized_text_pkl_path = Path("ldcc_tokenized_text_list.pkl").resolve()
        word_freqs_path = Path("ldcc_word_freqs.json").resolve()

        text_classification_dataset = TextClassificationDataset.create(
            labeled_tokenized_text_pkl_path, word_freqs_path
        )
        print(*list(text_classification_dataset)[:10], sep="\n")
    test2()
