from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json
import itertools
from subprocess import run

from libs.text_pair_dataset import get_tokenized_text_list


def create_word_freqs(file_path: Path) -> dict:
    tokenized_text_list = get_tokenized_text_list(file_path)
    counter = Counter(list(itertools.chain.from_iterable(tokenized_text_list)))

    return OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


def main():
    # データセットのアドレスと保存ファイル名
    url_base = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
    file_names = ["train.de.gz", "train.en.gz", "val.de.gz", "val.en.gz"]
    base_path = Path(__file__).resolve().parent / "multi30k_dataset"
    base_path.mkdir(exist_ok=True, parents=True)

    for file_name in file_names:
        # ダウンロード
        url = url_base + file_name
        zip_path = base_path / file_name
        print(f"download {file_name}")
        urlretrieve(url, zip_path)

        # zipファイルを展開する
        print(f"expand {file_name}")
        cmd = ["gzip", "-d", zip_path]
        run(cmd)

    # 単語ID辞書をjson形式で保存する
    print("create word frequency file...")
    word_freqs_de = create_word_freqs(base_path / "train.de")
    with (base_path / "word_freqs_de.json").open("w") as f:
        json.dump(word_freqs_de, f, indent=2, ensure_ascii=False)

    word_freqs_en = create_word_freqs(base_path / "train.en")
    with (base_path / "word_freqs_en.json").open("w") as f:
        json.dump(word_freqs_en, f, indent=2, ensure_ascii=False)

    print("done.")


if __name__ == "__main__":
    main()
