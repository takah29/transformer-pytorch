from pathlib import Path
import csv
import json

import torch
from torch import optim

from libs.text_pair_dataset import TextPairDataset
from libs.transformer import Transformer
from libs.translation_model_trainer import TranslationModelTrainer, TransformerLRScheduler


def get_instance(params: dict):
    transformer = Transformer(**params)
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = TransformerLRScheduler(optimizer, transformer.n_dim, warmup_steps=4000)

    return transformer, optimizer, lr_scheduler


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Learning the model.")
    parser.add_argument("dataset_dir", help="Dataset root directory path", type=str)

    args = parser.parse_args()

    base_path = Path(args.dataset_dir).resolve()

    # 指定したディレクトリが存在しない場合は終了する
    if not base_path.exists():
        print("Target directory does not exist.")
        return

    # パラメータ設定の読み込みと設定
    with (base_path / "settings.json").open("r") as f:
        settings = json.load(f)

    # 学習データセット作成
    src_txt_file_path = base_path / "src_train_texts.txt"
    tgt_txt_file_path = base_path / "tgt_train_texts.txt"
    src_word_freqs_path = base_path / "src_word_freqs.json"
    tgt_word_freqs_path = base_path / "tgt_word_freqs.json"
    src_min_freq = settings["min_freq"]["source"]
    tgt_min_freq = settings["min_freq"]["target"]

    train_dataset = TextPairDataset.create(
        src_txt_file_path,
        tgt_txt_file_path,
        src_word_freqs_path,
        tgt_word_freqs_path,
        src_min_freq,
        tgt_min_freq,
    )

    # 検証データセット作成
    src_val_txt_file_path = base_path / "src_val_texts.txt"
    tgt_val_txt_file_path = base_path / "tgt_val_texts.txt"
    valid_dataset = TextPairDataset.create(
        src_val_txt_file_path,
        tgt_val_txt_file_path,
        src_word_freqs_path,
        tgt_word_freqs_path,
        src_min_freq,
        tgt_min_freq,
    )

    # GPUが使える場合は使う
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    enc_vocab_size, dec_vocab_size = train_dataset.get_vocab_size()
    settings["params"]["enc_vocab_size"] = enc_vocab_size
    settings["params"]["dec_vocab_size"] = dec_vocab_size

    # インスタンス作成
    model, optimizer, lr_scheduler = get_instance(settings["params"])

    # モデル保存パス
    save_path = base_path / "models"

    # Trainerの作成と学習の実行
    translation_model_trainer = TranslationModelTrainer(
        model, optimizer, lr_scheduler, device, train_dataset, valid_dataset, save_path
    )
    train_loss_list, valid_loss_list = translation_model_trainer.fit(**settings["training"])

    # Lossをcsvファイルに保存
    with (save_path / "loss.csv").open("w") as f:
        header = ["epoch", "train_loss", "valid_loss"]
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for epoch, row in enumerate(zip(train_loss_list, valid_loss_list), start=1):
            csv_writer.writerow((epoch,) + row)


if __name__ == "__main__":
    main()
