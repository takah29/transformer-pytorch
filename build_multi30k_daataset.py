from pathlib import Path
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json
import itertools
from subprocess import run

from libs.text_encoder import get_tokenized_text_list


def get_datasets(train_file_path: Path, val_file_path: Path, lang="en"):
    train_tokenized_texts = get_tokenized_text_list(train_file_path, lang)
    val_tokenized_texts = get_tokenized_text_list(val_file_path, lang)
    counter = Counter(list(itertools.chain.from_iterable(train_tokenized_texts)))

    results = {
        "train_texts": train_tokenized_texts,
        "val_texts": val_tokenized_texts,
        "word_freqs": OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True)),
    }
    return results


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

    print("create de files...")
    results = get_datasets(base_path / "train.de", base_path / "val.de", lang="de")

    with (base_path / "de_train_texts.txt").open("w") as f:
        for tokenized_text in results["train_texts"]:
            f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "de_val_texts.txt").open("w") as f:
        for tokenized_text in results["val_texts"]:
            f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "de_word_freqs.json").open("w") as f:
        json.dump(results["word_freqs"], f, indent=2, ensure_ascii=False)

    print("create en files...")
    results = get_datasets(base_path / "train.en", base_path / "val.en", lang="en")

    with (base_path / "en_train_texts.txt").open("w") as f:
        for tokenized_text in results["train_texts"]:
            f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "en_val_texts.txt").open("w") as f:
        for tokenized_text in results["val_texts"]:
            f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "en_word_freqs.json").open("w") as f:
        json.dump(results["word_freqs"], f, indent=2, ensure_ascii=False)

    print("done.")


if __name__ == "__main__":
    main()
