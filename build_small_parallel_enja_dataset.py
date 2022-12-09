import itertools
import json
from collections import Counter, OrderedDict
from pathlib import Path
from shutil import unpack_archive
from urllib.request import urlretrieve

from libs.text_encoder import get_tokenized_text_list


def get_datasets(train_file_path: Path, val_file_path: Path, test_file_path: Path, lang: str):
    train_tokenized_texts = get_tokenized_text_list(train_file_path, lang)
    val_tokenized_texts = get_tokenized_text_list(val_file_path, lang)
    test_tokenized_texts = get_tokenized_text_list(test_file_path, lang)
    counter = Counter(list(itertools.chain.from_iterable(train_tokenized_texts)))

    results = {
        "train_texts": train_tokenized_texts,
        "val_texts": val_tokenized_texts,
        "test_texts": test_tokenized_texts,
        "word_freqs": OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True)),
    }
    return results


def write_parameter_settings(base_path: Path):
    settings = {
        "params": {
            "n_dim": 128,
            "hidden_dim": 500,
            "n_enc_blocks": 2,
            "n_dec_blocks": 2,
            "head_num": 8,
            "dropout_rate": 0.3,
        },
        "training": {
            "batch_size": 128,
            "num_epoch": 20,
        },
        "min_freq": {
            "source": 1,
            "target": 1,
        },
    }
    with (base_path / "settings.json").open("w") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


def main():
    # データセットのアドレスと保存ファイル名
    url = "https://github.com/odashi/small_parallel_enja/archive/refs/heads/master.zip"
    file_name = "small_parallel_enja-master.zip"
    base_path = Path(__file__).resolve().parent / "small_parallel_enja_dataset"
    base_path.mkdir(exist_ok=True, parents=True)

    zip_path = base_path / file_name

    # 存在しない場合にファイルをダウンロードする
    if not zip_path.exists():
        print(f"download {file_name}")
        urlretrieve(url, zip_path)

    # zipファイルを展開する
    print(f"expand {file_name}")
    unpack_archive(zip_path, base_path)
    dataset_path = base_path / "small_parallel_enja-master"

    print("create source files...")
    results = get_datasets(
        dataset_path / "train.en", dataset_path / "dev.en", dataset_path / "test.en", lang="spaced"
    )

    for key in ["train_texts", "val_texts", "test_texts"]:
        with (base_path / f"src_{key}.txt").open("w") as f:
            for tokenized_text in results[key]:
                f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "src_word_freqs.json").open("w") as f:
        json.dump(results["word_freqs"], f, indent=2, ensure_ascii=False)

    print("create target files...")
    results = get_datasets(
        dataset_path / "train.ja", dataset_path / "dev.ja", dataset_path / "test.ja", lang="spaced"
    )

    for key in ["train_texts", "val_texts", "test_texts"]:
        with (base_path / f"tgt_{key}.txt").open("w") as f:
            for tokenized_text in results[key]:
                f.write(" ".join(tokenized_text) + "\n")

    with (base_path / "tgt_word_freqs.json").open("w") as f:
        json.dump(results["word_freqs"], f, indent=2, ensure_ascii=False)

    print("write parameter settings")
    write_parameter_settings(base_path)

    print("done.")


if __name__ == "__main__":
    main()
