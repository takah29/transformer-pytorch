from pathlib import Path
import csv

import torch
from torch import optim

from libs.text_pair_dataset import TextPairDataset
from libs.transformer import Transformer
from libs.translation_model_trainer import TranslationModelTrainer


def get_instance(enc_vocab_size, dec_vocab_size):
    transformer = Transformer.create(enc_vocab_size, dec_vocab_size)
    # optimizer = optim.Adam(transformer.parameters(), lr=0.0, betas=(0.9, 0.98), eps=10e-9)
    optimizer = optim.Adam(transformer.parameters())

    return transformer, optimizer


def main():
    # 事前にbuild_multi30k_word_freqs.pyを実行してデータセットのダウンロードと頻度辞書の作成を行っておく
    base_path = Path("./small_parallel_enja_dataset").resolve()

    # 学習データセット作成
    de_txt_file_path = base_path / "en_train_texts.txt"
    en_txt_file_path = base_path / "ja_train_texts.txt"
    de_word_freqs_path = base_path / "en_word_freqs.json"
    en_word_freqs_path = base_path / "ja_word_freqs.json"
    train_dataset = TextPairDataset.create(
        de_txt_file_path, en_txt_file_path, de_word_freqs_path, en_word_freqs_path
    )

    # 検証データセット作成
    de_val_txt_file_path = base_path / "en_val_texts.txt"
    en_val_txt_file_path = base_path / "ja_val_texts.txt"
    valid_dataset = TextPairDataset.create(
        de_val_txt_file_path, en_val_txt_file_path, de_word_freqs_path, en_word_freqs_path
    )

    # ネットワークパラメータ定義
    enc_vocab_size, dec_vocab_size = train_dataset.get_vocab_size()

    # GPUが使える場合は使う
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # インスタンス作成
    model, optimizer = get_instance(enc_vocab_size, dec_vocab_size)

    # モデル保存パス
    save_path = Path(__file__).resolve().parent / "models"

    # Trainerの作成と学習の実行
    translation_model_trainer = TranslationModelTrainer(
        model, optimizer, None, device, train_dataset, valid_dataset, save_path
    )
    train_loss_list, valid_loss_list = translation_model_trainer.fit(batch_size=128, num_epoch=20)

    # Lossをcsvファイルに保存
    with (save_path / "loss.csv").open("w") as f:
        header = ["epoch", "train_loss", "valid_loss"]
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for epoch, row in enumerate(zip(train_loss_list, valid_loss_list), start=1):
            csv_writer.writerow((epoch,) + row)


if __name__ == "__main__":
    main()
