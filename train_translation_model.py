from pathlib import Path
import csv

import torch
from torch import optim

from libs.text_pair_dataset import TextPairDataset
from libs.transformer import Transformer
from libs.translation_model_trainer import TranslationModelTrainer


def get_instance(params):
    transformer = Transformer(**params)
    # optimizer = optim.Adam(transformer.parameters(), lr=0.0, betas=(0.9, 0.98), eps=10e-9)
    optimizer = optim.Adam(transformer.parameters())

    return transformer, optimizer


def main():
    # 事前にbuild_word_freqs.pyを実行してデータセットのダウンロードと頻度辞書の作成を行っておく
    base_path = Path("./small_parallel_enja_dataset").resolve()

    # 学習データセット作成
    en_txt_file_path = base_path / "small_parallel_enja-master" / "train.en"
    ja_txt_file_path = base_path / "small_parallel_enja-master" / "train.ja"
    en_word_freqs_path = base_path / "word_freqs_en.json"
    ja_word_freqs_path = base_path / "word_freqs_ja.json"
    train_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )

    # 検証データセット作成
    en_val_txt_file_path = base_path / "small_parallel_enja-master" / "dev.en"
    ja_val_txt_file_path = base_path / "small_parallel_enja-master" / "dev.ja"
    valid_dataset = TextPairDataset.create(
        en_val_txt_file_path, ja_val_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )

    # ネットワークパラメータ定義
    enc_vocab_size, dec_vocab_size = train_dataset.get_vocab_size()
    params = {
        "enc_vocab_size": enc_vocab_size,
        "dec_vocab_size": dec_vocab_size,
        "n_dim": 240,
        "hidden_dim": 100,
        "n_enc_blocks": 2,
        "n_dec_blocks": 2,
        "head_num": 8,
        "dropout_rate": 0.1,
    }

    # GPUが使える場合は使う
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # インスタンス作成
    model, optimizer = get_instance(params)

    # モデル保存パス
    save_path = Path(__file__).resolve().parent / "models"

    # Trainerの作成と学習の実行
    translation_model_trainer = TranslationModelTrainer(
        model, optimizer, None, device, train_dataset, valid_dataset, save_path
    )
    train_loss_list, valid_loss_list = translation_model_trainer.fit(batch_size=128, num_epoch=2)

    # Lossをcsvファイルに保存
    with (save_path / "loss.csv").open("w") as f:
        header = ["train_loss", "valid_loss"]
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for epoch, row in enumerate(zip(train_loss_list, valid_loss_list), start=1):
            csv_writer.writerow((epoch,) + row)


if __name__ == "__main__":
    main()
