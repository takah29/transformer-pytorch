from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class TransformerLRScheduler(_LRScheduler):
    def __init__(self, optimizer, n_dim, warmup_steps):
        self._n_dim = n_dim
        self._warmup_steps = warmup_steps

        self.optimizer = optimizer
        self._steps = 0

    def step(self):
        self._steps += 1

        self.lr = self._n_dim * min(
            self._steps ** (-0.5), self._steps * self._warmup_steps ** (-1.5)
        )
        self.set_lr(self.optimizer, self.lr)

        return self.lr

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g["lr"] = lr


class TranslationModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        device,
        train_dataset,
        valid_dataset=None,
    ):
        self._device = device
        self._model = model.to(self._device)

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        _, self._target_vocab_size = train_dataset.get_vocab_size()

        self._criterion = nn.CrossEntropyLoss(ignore_index=self._train_dataset.get_pad_id()[0])

    def fit(self, batch_size, num_epoch):
        train_loss = valid_loss = None

        train_data_loader = DataLoader(
            self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )
        train_loss_list = []

        if self._valid_dataset is not None:
            valid_data_loader = DataLoader(
                self._valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4
            )
            valid_loss_list = []

        for i in range(num_epoch):
            print("epoch: ", i + 1)

            self._model.train()
            train_loss = self._run_epoch(train_data_loader, is_train=True)
            train_loss_list.append(train_loss)

            if self._valid_dataset is not None:
                self._model.eval()
                valid_loss = self._run_epoch(valid_data_loader, is_train=True)
                valid_loss_list.append(valid_loss)

            print(f"train loss: {train_loss}, valid loss: {valid_loss}")

        return train_loss_list, valid_loss_list

    def _run_epoch(self, data_loader, is_train):
        s = 0.0
        for batch in tqdm(data_loader):
            # print(i + 1, x.size(), t.size())
            enc_input_text = batch["enc_input"]["text"].to(self._device)
            dec_input_text = batch["dec_input"]["text"].to(self._device)
            enc_input_mask = batch["enc_input"]["mask"].to(self._device)
            dec_input_mask = batch["dec_input"]["mask"].to(self._device)

            y = self._model(
                enc_input_text,
                dec_input_text,
                enc_input_mask,
                dec_input_mask,
            )
            t = batch["dec_target"].to(self._device)

            # print(y.argmax(-1).shape, t.shape)
            loss = self._criterion(y.reshape(-1, y.shape[-1]), t.reshape(-1))
            s += loss.item()

            if is_train:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

        return s / len(data_loader.dataset)


def get_instance(params):
    transformer = Transformer(**params)
    # optimizer = optim.Adam(transformer.parameters(), lr=0.0, betas=(0.9, 0.98), eps=10e-9)
    optimizer = optim.Adam(transformer.parameters())
    lr_scheduler = TransformerLRScheduler(optimizer, params["n_dim"], warmup_steps=4000)

    return transformer, optimizer, lr_scheduler


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from text_pair_dataset import TextPairDataset
    from transformer import Transformer

    # 事前にbuild_word_freqs.pyを実行してデータセットのダウンロードと頻度辞書の作成を行っておく
    base_path = Path("../dataset/small_parallel_enja-master").resolve()
    en_txt_file_path = base_path / "train.en"
    ja_txt_file_path = base_path / "train.ja"
    en_word_freqs_path = Path("../word_freqs_en.json").resolve()
    ja_word_freqs_path = Path("../word_freqs_ja.json").resolve()
    train_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )

    en_val_txt_file_path = base_path / "dev.en"
    ja_val_txt_file_path = base_path / "dev.ja"
    valid_dataset = TextPairDataset.create(
        en_val_txt_file_path, ja_val_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )

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
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model, optimizer, scheduler = get_instance(params)
    translation_model_trainer = TranslationModelTrainer(
        model, optimizer, None, device, train_dataset, valid_dataset
    )
    train_loss_list, valid_loss_list = translation_model_trainer.fit(batch_size=128, num_epoch=10)

    plt.plot(train_loss_list)
    plt.plot(valid_loss_list)

    plt.show()
