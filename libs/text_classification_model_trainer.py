import torch
from torch import nn, optim
from torch.utils.data import DataLoader


class TextClassificationModelTrainer:
    def __init__(self, model, optimizer, device, train_dataset, valid_dataset=None, lr=0.05):
        self._device = device
        self._model = model.to(self._device)

        self._optimizer = optimizer(self._model.parameters(), lr=lr)
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._target_vocab_size = train_dataset.get_vocab_size()

    def fit(self, batch_size, num_epoch):
        criterion = nn.CrossEntropyLoss()
        train_data_loader = DataLoader(
            self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

        loss_list = []
        self._model.train()
        for i in range(num_epoch):
            print("epoch: ", i + 1)
            for i, batch in enumerate(train_data_loader):
                input_text = batch["input"]["text"].to(self._device)
                input_mask = batch["input"]["mask"].to(self._device)

                y = self._model(
                    input_text,
                    input_mask,
                )
                t = batch["target"].to(self._device)

                loss = criterion(y, t)
                print(loss.item())
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                loss_list.append(loss.item())

        return loss_list


if __name__ == "__main__":
    from pathlib import Path

    from text_classification_dataset import TextClassificationDataset
    from transformer import TransformerClassifier

    # 事前にbuild_ldcc_word_freqs.pyを実行してデータセットのダウンロードと頻度辞書の作成を行っておく
    base_path = Path(__file__).resolve().parents[1] / "ldcc_dataset"
    labeled_tokenized_text_pkl_path = base_path / "ldcc_tokenized_text_list.pkl"
    word_freqs_path = base_path / "ldcc_word_freqs.json"

    text_classification_dataset = TextClassificationDataset.create(
        labeled_tokenized_text_pkl_path, word_freqs_path
    )

    vocab_size = text_classification_dataset.get_vocab_size()
    n_classes = text_classification_dataset.get_class_num()
    params = {
        "n_classes": n_classes,
        "vocab_size": vocab_size,
        "n_dim": 128,
        "hidden_dim": 64,
        "n_enc_blocks": 1,
        "head_num": 1,
    }
    transformer_classifier = TransformerClassifier(**params)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    text_classification_model_trainer = TextClassificationModelTrainer(
        transformer_classifier, optim.Adam, device, text_classification_dataset, None, 1.7e-3
    )
    loss_list = text_classification_model_trainer.fit(batch_size=1000, num_epoch=2)
