from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve
from collections import Counter, OrderedDict
import json
import itertools

from tqdm import tqdm
from libs.text_encoder import create_tokenizer


def get_tokenzied_texts(file_path):
    print(f"processing {file_path.name} file...")
    en_tokenizer = create_tokenizer("en")
    ja_tokenizer = create_tokenizer("ja")

    en_tokenized_texts = []
    ja_tokenized_texts = []
    with file_path.open("r") as f:
        for line in tqdm(f.readlines()):
            en_line, ja_line = line.strip().split("\t")
            en_tokenized_texts.append(en_tokenizer(en_line))
            ja_tokenized_texts.append(ja_tokenizer(ja_line))

    return en_tokenized_texts, ja_tokenized_texts


def get_datasets(train_file_path: Path, dev_file_path: Path, test_file_path: Path):
    en_train_tokenized_texts, ja_train_tokenized_texts = get_tokenzied_texts(train_file_path)
    en_val_tokenized_texts, ja_val_tokenized_texts = get_tokenzied_texts(dev_file_path)
    en_test_tokenized_texts, ja_test_tokenized_texts = get_tokenzied_texts(test_file_path)

    en_counter = Counter(list(itertools.chain.from_iterable(en_train_tokenized_texts)))
    ja_counter = Counter(list(itertools.chain.from_iterable(ja_train_tokenized_texts)))

    en_results = {
        "train_texts": en_train_tokenized_texts,
        "val_texts": en_val_tokenized_texts,
        "test_texts": en_test_tokenized_texts,
        "word_freqs": OrderedDict(sorted(en_counter.items(), key=lambda x: x[1], reverse=True)),
    }
    ja_results = {
        "train_texts": ja_train_tokenized_texts,
        "val_texts": ja_val_tokenized_texts,
        "test_texts": ja_test_tokenized_texts,
        "word_freqs": OrderedDict(sorted(ja_counter.items(), key=lambda x: x[1], reverse=True)),
    }

    return en_results, ja_results


def write_parameter_settings(base_path: Path):
    settings = {
        "params": {
            "n_dim": 128,
            "hidden_dim": 500,
            "n_enc_blocks": 4,
            "n_dec_blocks": 4,
            "head_num": 8,
            "dropout_rate": 0.3,
        },
        "training": {
            "batch_size": 128,
            "num_epoch": 20,
        },
    }
    with (base_path / "settings.json").open("w") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def main():
    # データセットのアドレスと保存ファイル名
    url = "https://nlp.stanford.edu/projects/jesc/data/split.tar.gz"
    file_name = "split.tar.gz"
    base_path = Path(__file__).resolve().parent / "jesc_dataset"
    base_path.mkdir(exist_ok=True, parents=True)

    zip_path = base_path / file_name

    # 存在しない場合にファイルをダウンロードする
    if not zip_path.exists():
        print(f"download {file_name}")
        urlretrieve(url, zip_path)

    # zipファイルを展開する
    print(f"expand {file_name}")
    unpack_archive(zip_path, base_path)

    # トークナイズ、単語頻度データ作成
    dataset_path = base_path / "split"
    en_results, ja_results = get_datasets(
        dataset_path / "train", dataset_path / "dev", dataset_path / "test"
    )

    # 英語ファイルの書き込み
    print("create source files...")

    for key in ["train_texts", "val_texts", "test_texts"]:
        with (base_path / f"src_{key}.txt").open("w") as f:
            for tokenized_text in en_results[key]:
                f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "src_word_freqs.json").open("w") as f:
        json.dump(en_results["word_freqs"], f, indent=2, ensure_ascii=False)

    # 日本語ファイルの書き込み
    print("create target files...")

    for key in ["train_texts", "val_texts", "test_texts"]:
        with (base_path / f"tgt_{key}.txt").open("w") as f:
            for tokenized_text in ja_results[key]:
                f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "tgt_word_freqs.json").open("w") as f:
        json.dump(ja_results["word_freqs"], f, indent=2, ensure_ascii=False)

    print("write parameter settings")
    write_parameter_settings(base_path)

    print("done.")


if __name__ == "__main__":
    main()
