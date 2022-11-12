import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F


class TranslationModelTrainer:
    def __init__(self, model, optimizer, dataset, device, lr=0.05):
        self._device = device
        self._model = model.to(self._device)

        self._optimizer = optimizer(self._model.parameters(), lr=lr)
        self._dataset = dataset
        _, self._target_vocab_size = dataset.get_vocab_size()

    # def _collate_fn(self, batch):
    #     print(batch)
    #     batch["dec_target"] = F.one_hot(batch["dec_target"], num_classes=self._target_vocab_size)
    #     return batch

    def fit(self, batch_size, num_epoch):
        criterion = nn.CrossEntropyLoss()
        data_loader = DataLoader(self._dataset, batch_size=batch_size, shuffle=True)

        loss_list = []
        s = 0.0
        for epoch in range(num_epoch):
            print("epoch: ", epoch)
            s = 0.0
            for i, batch in enumerate(data_loader):
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
                t = (
                    F.one_hot(batch["dec_target"], num_classes=self._target_vocab_size)
                    .to(torch.float32)
                    .to(self._device)
                )
                # print(
                #     f"batch: {i+1}",
                #     y,
                #     t.shape,
                # )
                loss = criterion(y, t)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                s += loss.item()
            loss_list.append(s)
            print(s)


if __name__ == "__main__":
    from pathlib import Path
    from dataset import TextPairDataset
    from transformer import Transformer

    # 事前にbuild_word_freqs.pyを実行してデータセットのダウンロードと頻度辞書の作成を行っておく
    en_txt_file_path = Path("dataset/small_parallel_enja-master/dev.en").resolve()
    ja_txt_file_path = Path("dataset/small_parallel_enja-master/dev.ja").resolve()
    en_word_freqs_path = Path("word_freqs_en.json").resolve()
    ja_word_freqs_path = Path("word_freqs_ja.json").resolve()

    text_pair_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )

    enc_vocab_size, dec_vocab_size = text_pair_dataset.get_vocab_size()
    params = {
        "enc_vocab_size": enc_vocab_size,
        "dec_vocab_size": dec_vocab_size,
        "n_dim": 64,
        "hidden_dim": 32,
        "token_size": 40,
        "n_enc_blocks": 3,
        "n_dec_blocks": 2,
        "head_num": 4,
    }
    transformer = Transformer(**params)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    translation_model_trainer = TranslationModelTrainer(
        transformer, optim.Adam, text_pair_dataset, device, 0.05
    )
    translation_model_trainer.fit(batch_size=100, num_epoch=10)
