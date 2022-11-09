from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json

from torchtext.vocab import vocab


def create_vocab(file_path: Path) -> dict:
    word_list = []
    with file_path.open("r") as f:
        for line in f:
            word_list.extend(line.strip().split())

    counter = Counter(word_list)
    return vocab(
        OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True)),
        specials=["<pad>", "<unk>", "<bos>", "<eos>"],
    )


def main():
    # データセットのアドレスと保存ファイル名
    url = "https://github.com/odashi/small_parallel_enja/archive/refs/heads/master.zip"
    file_name = "small_parallel_enja-master.zip"
    base_path = Path(__file__).resolve().parent
    zip_path = base_path / file_name

    # 存在しない場合にファイルをダウンロードする
    if not zip_path.exists():
        print(f"download {file_name}")
        urlretrieve(url, zip_path)

    # zipファイルを展開する
    unpack_archive(zip_path, base_path / "dataset")
    dataset_path = base_path / "dataset" / "small_parallel_enja-master"

    # 単語ID辞書をjson形式で保存する
    vocab_ja = create_vocab(dataset_path / "train.ja")
    with (base_path / "word_id_dict_ja.json").open("w") as f:
        json.dump(vocab_ja.get_stoi(), f, indent=2, ensure_ascii=False)

    vocab_en = create_vocab(dataset_path / "train.en")
    with (base_path / "word_id_dict_en.json").open("w") as f:
        json.dump(vocab_en.get_stoi(), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
