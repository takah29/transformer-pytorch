from pathlib import Path
import csv

import torch
from torch import optim

from libs.text_pair_dataset import TextPairDataset
from libs.transformer import Transformer
from libs.translation_model_trainer import TranslationModelTrainer, TransformerLRScheduler


def get_instance(enc_vocab_size, dec_vocab_size):
    transformer = Transformer.create(enc_vocab_size, dec_vocab_size)
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    # optimizer = optim.Adam(transformer.parameters())
    lr_scheduler = TransformerLRScheduler(optimizer, transformer.n_dim, warmup_steps=4000)

    return transformer, optimizer, lr_scheduler


def main():
    base_path = Path("./small_parallel_enja_dataset").resolve()

    # 学習データセット作成
    src_txt_file_path = base_path / "src_train_texts.txt"
    tgt_txt_file_path = base_path / "tgt_train_texts.txt"
    src_word_freqs_path = base_path / "src_word_freqs.json"
    tgt_word_freqs_path = base_path / "tgt_word_freqs.json"
    train_dataset = TextPairDataset.create(
        src_txt_file_path, tgt_txt_file_path, src_word_freqs_path, tgt_word_freqs_path
    )

    # 検証データセット作成
    src_val_txt_file_path = base_path / "src_val_texts.txt"
    tgt_val_txt_file_path = base_path / "tgt_val_texts.txt"
    valid_dataset = TextPairDataset.create(
        src_val_txt_file_path, tgt_val_txt_file_path, src_word_freqs_path, tgt_word_freqs_path
    )

    # ネットワークパラメータ定義
    enc_vocab_size, dec_vocab_size = train_dataset.get_vocab_size()

    # GPUが使える場合は使う
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # インスタンス作成
    model, optimizer, lr_scheduler = get_instance(enc_vocab_size, dec_vocab_size)

    # モデル保存パス
    save_path = Path(__file__).resolve().parent / "models"

    # Trainerの作成と学習の実行
    translation_model_trainer = TranslationModelTrainer(
        model, optimizer, lr_scheduler, device, train_dataset, valid_dataset, save_path
    )
    train_loss_list, valid_loss_list = translation_model_trainer.fit(batch_size=128, num_epoch=2)

    # Lossをcsvファイルに保存
    with (save_path / "loss.csv").open("w") as f:
        header = ["epoch", "train_loss", "valid_loss"]
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
        for epoch, row in enumerate(zip(train_loss_list, valid_loss_list), start=1):
            csv_writer.writerow((epoch,) + row)


if __name__ == "__main__":
    main()
