from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json
import itertools
import pickle

from libs.text_encoder import create_tokenizer


def get_ldcc_labeled_tokenized_text_list(base_dir_path):
    """ldccデータセットを単語リスト化したデータを返す"""
    # フォルダ名: クラス番号
    dir_names = [
        "dokujo-tsushin",
        "kaden-channel",
        "movie-enter",
        "smax",
        "topic-news",
        "it-life-hack",
        "livedoor-homme",
        "peachy",
        "sports-watch",
    ]

    tokenizer = create_tokenizer(lang="ja")
    results = {"tokenized_text_list": [], "class_label_list": [], "class_name_list": dir_names}
    for class_num, dir_name in enumerate(dir_names):
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
                tokenized_text = tokenizer(line)
                results["tokenized_text_list"].append(list(tokenized_text))
                results["class_label_list"].append(class_num)

    return results


def create_word_freqs(tokenized_text_list) -> dict:
    counter = Counter(list(itertools.chain.from_iterable(tokenized_text_list)))

    return OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))


def main():
    # データセットのアドレスと保存ファイル名
    url = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"
    file_name = "ldcc-20140209.tar.gz"
    base_path = Path(__file__).resolve().parent / "ldcc_dataset"
    base_path.mkdir(exist_ok=True, parents=True)

    archive_path = base_path / file_name

    # 存在しない場合にファイルをダウンロードする
    if not archive_path.exists():
        print(f"download {file_name}")
        urlretrieve(url, archive_path)

    # 圧縮ファイルを展開する
    print(f"expand {file_name}")
    unpack_archive(archive_path, base_path, format="gztar")

    # 単語ID辞書をjson形式で保存する
    print("create word frequency file...")
    labeled_tokenized_text_list = get_ldcc_labeled_tokenized_text_list(base_path / "text")
    with (base_path / "ldcc_tokenized_text_list.pkl").open("wb") as f:
        pickle.dump(labeled_tokenized_text_list, f)

    word_freqs = create_word_freqs(labeled_tokenized_text_list["tokenized_text_list"])
    with (base_path / "ldcc_word_freqs.json").open("w") as f:
        json.dump(word_freqs, f, indent=2, ensure_ascii=False)

    print("done.")


if __name__ == "__main__":
    main()
