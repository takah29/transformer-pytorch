from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json
import itertools
import pickle

from janome.tokenizer import Tokenizer


def get_ldcc_labeled_tokenized_text_list(base_dir_path):
    """ldccデータセットを単語リスト化したデータを返す"""
    # フォルダ名: クラス番号
    dir_names = {
        "dokujo-tsushin": 0,
        "kaden-channel": 1,
        "movie-enter": 2,
        "smax": 3,
        "topic-news": 4,
        "it-life-hack": 5,
        "livedoor-homme": 6,
        "peachy": 7,
        "sports-watch": 8,
    }
    tokenizer = Tokenizer()
    data = {"tokenized_text_list": [], "class_label_list": [], "class_name_list": []}
    for dir_name, class_num in dir_names.items():
        dir_path = base_dir_path / dir_name
        print("process of", dir_path.name)

        for filepath in [
            filepath for filepath in dir_path.iterdir() if filepath.name != "LICENSE.txt"
        ]:
            with filepath.open("r") as f:
                lines = [line.strip() for line in f.readlines()]

            for line in lines[2:]:  # 最初の2行はスキップ
                if line == "":
                    continue
                tokenized_text = tokenizer.tokenize(line, wakati=True)
                data["tokenized_text_list"].append(list(tokenized_text))
                data["class_label_list"].append(class_num)
                data["class_name_list"].append(dir_name)

    return data


def create_word_freqs(tokenized_text_list) -> dict:
    counter = Counter(list(itertools.chain.from_iterable(tokenized_text_list)))

    return OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


def main():
    # データセットのアドレスと保存ファイル名
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    file_name = "ldcc-20140209.tar.gz"
    base_path = Path(__file__).resolve().parent
    archive_path = base_path / file_name

    # 存在しない場合にファイルをダウンロードする
    if not archive_path.exists():
        print(f"download {file_name}")
        urlretrieve(url, archive_path)

    # 圧縮ファイルを展開する
    print(f"expand {file_name}")
    unpack_archive(archive_path, base_path / "dataset", format="gztar")
    dataset_path = base_path / "dataset"

    # 単語ID辞書をjson形式で保存する
    print("create word frequency file...")
    labeled_tokenized_text_list = get_ldcc_labeled_tokenized_text_list(dataset_path / "text")
    with open("ldcc_tokenized_text_list.pkl", "wb") as f:
        pickle.dump(labeled_tokenized_text_list, f)

    word_freqs = create_word_freqs(labeled_tokenized_text_list["tokenized_text_list"])
    with (base_path / "ldcc_word_freqs.json").open("w") as f:
        json.dump(word_freqs, f, indent=2, ensure_ascii=False)

    print("done.")


if __name__ == "__main__":
    main()
