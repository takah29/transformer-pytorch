from pathlib import Path
import json

from torch.utils.data import Dataset
import torchtext.transforms as T
from torchtext.vocab import vocab


def get_tokenized_text_list(file_path):
    tokenized_text_list = []
    with file_path.open("r") as f:
        for line in f:
            tokenized_text_list.append(line.strip().split())

    return tokenized_text_list


def get_vocab(word_id_dict_path):
    with open(word_id_dict_path, "r") as f:
        en_word_id_dict = json.load(f)

    return vocab(en_word_id_dict)


def get_transform(word_count, vocab_data):
    text_transform = T.Sequential(
        T.VocabTransform(vocab_data),  # トークンに変換
        T.Truncate(word_count),  # word_countを超過したデータを切り捨てる
        T.AddToken(token=vocab_data["<bos>"], begin=True),  # 先頭に'<bos>追加
        T.AddToken(token=vocab_data["<eos>"], begin=False),  # 終端に'<eos>'追加
        T.ToTensor(),  # テンソルに変換
        T.PadTransform(word_count + 2, vocab_data["<pad>"]),  # word_countに満たない文章を'<pad>'で埋める
    )

    return text_transform


class TextPairDataset(Dataset):
    def __init__(
        self, tokenized_text_list_1, tokenized_text_list_2, text_transform_1, text_transform_2
    ):
        super().__init__()

        self.tokenized_text_list_1 = tokenized_text_list_1
        self.tokenized_text_list_2 = tokenized_text_list_2

        self.text_transform_1 = text_transform_1
        self.text_transform_2 = text_transform_2

        # 2つのtextリストのうち要素数が少ない方のリストに合わせる。超過した要素は無視する
        self.n_data = min(len(self.tokenized_text_list_1), len(self.tokenized_text_list_2))

    def __getitem__(self, i):
        enc_input = self.tokenized_text_list_1[i]
        enc_input = self.text_transform_1([enc_input]).squeeze()

        target = self.tokenized_text_list_2[i]
        target = self.text_transform_2([target]).squeeze()

        dec_input = target[:-1]
        dec_target = target[1:]  # 右に1つずらす
        data = {"enc_input": enc_input, "dec_input": dec_input, "dec_target": dec_target}

        return data

    def __len__(self):
        return self.n_data

    @staticmethod
    def create(en_txt_file_path, ja_txt_file_path, en_word_id_dict_path, ja_word_id_dict_path):
        # 文章をトークン化してリスト化したデータを作成
        en_tokenized_text_list = get_tokenized_text_list(en_txt_file_path)
        ja_tokenized_text_list = get_tokenized_text_list(ja_txt_file_path)

        # 1文あたりの単語数
        word_count = 64
        en_vocab = get_vocab(en_word_id_dict_path)
        en_vocab.set_default_index(en_vocab["<unk>"])
        ja_vocab = get_vocab(ja_word_id_dict_path)
        ja_vocab.set_default_index(ja_vocab["<unk>"])

        en_text_transform = get_transform(word_count, en_vocab)
        ja_text_transform = get_transform(word_count, ja_vocab)

        return TextPairDataset(
            en_tokenized_text_list, ja_tokenized_text_list, en_text_transform, ja_text_transform
        )


if __name__ == "__main__":
    # 事前にbuild_word_id_dict.pyを実行してデータセットのダウンロードと単語辞書の作成を行っておく
    en_txt_file_path = Path("dataset/small_parallel_enja-master/dev.en").resolve()
    ja_txt_file_path = Path("dataset/small_parallel_enja-master/dev.ja").resolve()
    en_word_id_dict_path = Path("word_id_dict_en.json").resolve()
    ja_word_id_dict_path = Path("word_id_dict_ja.json").resolve()

    text_pair_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_id_dict_path, ja_word_id_dict_path
    )
    print(*list(text_pair_dataset)[:10], sep="\n")
