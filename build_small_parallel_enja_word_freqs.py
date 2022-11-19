from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json
import itertools

from dataset import get_tokenized_text_list


def create_word_freqs(file_path: Path) -> dict:
    tokenized_text_list = get_tokenized_text_list(file_path)
    counter = Counter(list(itertools.chain.from_iterable(tokenized_text_list)))

    return OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


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
    print(f"expand {file_name}")
    unpack_archive(zip_path, base_path / "dataset")
    dataset_path = base_path / "dataset" / "small_parallel_enja-master"

    # 単語ID辞書をjson形式で保存する
    print("create word frequency file...")
    word_freqs_ja = create_word_freqs(dataset_path / "train.ja")
    with (base_path / "word_freqs_ja.json").open("w") as f:
        json.dump(word_freqs_ja, f, indent=2, ensure_ascii=False)

    word_freqs_en = create_word_freqs(dataset_path / "train.en")
    with (base_path / "word_freqs_en.json").open("w") as f:
        json.dump(word_freqs_en, f, indent=2, ensure_ascii=False)

    print("done.")


if __name__ == "__main__":
    main()
