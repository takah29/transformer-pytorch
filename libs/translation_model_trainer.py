from pathlib import Path

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class TranslationLoss(nn.Module):
    def __init__(self, pad_id, label_smoothing=0.0):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1).long()
        return self.loss_func(logits, labels)


class TransformerLRScheduler(_LRScheduler):
    def __init__(self, optimizer, n_dim, warmup_steps):
        self._n_dim = n_dim
        self._warmup_steps = warmup_steps

        self.optimizer = optimizer
        self._steps = 0

    def step(self):
        self._steps += 1

        self.lr = self.calc_lr()
        self.set_lr(self.optimizer, self.lr)

        return self.lr

    def calc_lr(self):
        return self._n_dim ** (-0.5) * min(
            self._steps ** (-0.5), self._steps * self._warmup_steps ** (-1.5)
        )

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
        save_path=None,
    ):
        self._device = device
        self._model = model.to(self._device)

        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._save_path = save_path

        _, self._target_vocab_size = train_dataset.get_vocab_size()

        self._train_criterion = TranslationLoss(self._train_dataset.get_pad_id()[0], 0.1)
        if self._valid_dataset is not None:
            self._valid_criterion = TranslationLoss(self._valid_dataset.get_pad_id()[0], 0.0)
        else:
            self._valid_criterion = None

    def fit(self, batch_size: int, num_epoch: int):
        best_loss = float("inf")
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
                with torch.inference_mode():
                    valid_loss = self._run_epoch(valid_data_loader, is_train=False)
                valid_loss_list.append(valid_loss)

                # valid lossが一番低いモデルを保存する
                if self._save_path is not None and valid_loss < best_loss:
                    self._save_model("model.pth")
                    best_loss = valid_loss

            # 毎エポックごとにモデルを保存する
            if self._save_path:
                self._save_model(f"snapshot_epoch{i + 1:03}.pth")

            print(f"train loss: {train_loss}, valid loss: {valid_loss}")

        return train_loss_list, valid_loss_list

    def _run_epoch(self, data_loader, is_train: bool):
        s = 0.0
        pbar = tqdm(data_loader)
        for batch in pbar:
            enc_input_text = batch["enc_input"]["text"].to(self._device)
            dec_input_text = batch["dec_input"]["text"].to(self._device)
            enc_input_mask = batch["enc_input"]["mask"].to(self._device)
            dec_input_mask = batch["dec_input"]["mask"].to(self._device)
            t = batch["dec_target"].to(self._device)

            y = self._model(
                enc_input_text,
                dec_input_text,
                enc_input_mask,
                dec_input_mask,
            )
            if is_train:
                loss = self._train_criterion(y, t)
            else:
                loss = self._valid_criterion(y, t)

            s += loss.item()

            if is_train:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

            pbar.set_description(f"loss/iter: {loss.item():5.3f}, progress")

        return s / len(data_loader)

    def _save_model(self, filename):
        self._save_path.mkdir(exist_ok=True, parents=True)
        self._model.to("cpu")
        torch.save(self._model.state_dict(), self._save_path / filename)
        self._model.to(self._device)


def get_instance(params):
    transformer = Transformer(**params)
    optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1.0e-9)
    lr_scheduler = TransformerLRScheduler(optimizer, params["n_dim"], warmup_steps=4000)

    return transformer, optimizer, lr_scheduler


if __name__ == "__main__":
    from text_pair_dataset import TextPairDataset
    from transformer import Transformer

    base_path = Path(__file__).resolve().parents[1] / "multi30k_dataset"
    en_txt_file_path = base_path / "src_train_texts.txt"
    ja_txt_file_path = base_path / "tgt_train_texts.txt"
    en_word_freqs_path = base_path / "src_word_freqs.json"
    ja_word_freqs_path = base_path / "tgt_word_freqs.json"
    train_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )

    en_val_txt_file_path = base_path / "src_val_texts.txt"
    ja_val_txt_file_path = base_path / "tgt_val_texts.txt"
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
        model, optimizer, scheduler, device, train_dataset, valid_dataset
    )
    train_loss_list, valid_loss_list = translation_model_trainer.fit(batch_size=128, num_epoch=1)
