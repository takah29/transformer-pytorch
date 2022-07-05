import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from torch import nn, optim
from transformer import TransformerEncoder


# class TestTrainer(pl.LightningModule):
#     def __init__(self, input_dim, hidden_dim, num_labels, lr):
#         super.__init__()
#         self.save_hyperparameters()
#         self.model = TransformerEncoder(input_dim, hidden_dim)


#     def training_step(self, batch: dict):
#         output = self.model(batch)
#         loss = loss_function(output)
#         self.log("train_loss", loss)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         output = self.bert_sc(**batch)
#         val_loss = output.loss
#         self.log("val_loss", val_loss)

#     def test_step(self, batch, batch_idx):
#         labels = batch.pop("labels")
#         output = self.bert_sc(**batch)
#         labels_predicted = output.logits.argmax(-1)
#         num_correct = (labels_predicted == labels).sum().item()
#         accuracy = num_correct / labels.size(0)
#         self.log("accuracy", accuracy)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def train(model, optimizer, train_x, train_t):
    criterion = nn.CrossEntropyLoss()
    # dataset = torch.utils.data.TensorDataset(train_x, train_t)
    data_loader = DataLoader(list(zip(train_x, train_t)), batch_size=10, shuffle=True)

    loss_list = []
    for i, (x, t) in enumerate(data_loader):
        print(i+1, x.size(), t.size())
        y = model(x)
        # print(f"batch: {i+1}", x.size(), y, t)
        loss = criterion(y, t)
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()





def main():
    N = 1000
    x1 = torch.randint(0, 500, (N, 256))
    t1 = torch.zeros(N).to(torch.int64)
    x2 = torch.randint(500, 1000, (N, 256))
    t2 = torch.ones(N).to(torch.int64)

    X = torch.vstack((x1, x2))
    T = torch.hstack((t1, t2))

    x = torch.tensor(-75.0, requires_grad=True)
    y = torch.tensor(-10.0, requires_grad=True)
    params = [x, y]

    optimizer = optim.Adam(params)
    param = {
        "vocab_size": 1000,
        "n_dim": 100,
        "hidden_dim": 16,
        "token_size": 256
    }
    model = TransformerEncoder(**param)

    train(model, optimizer, X, T)


if __name__ == "__main__":
    main()

"""
class BertForSequenceClassification_pl(pl.LightningModule):
    def __init__(self, model_name, num_labels, lr):
        super().__init__()

        self.save_hyperparameters()

        self.bert_sc = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def training_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        loss = output.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.bert_sc(**batch)
        val_loss = output.loss
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        labels = batch.pop("labels")
        output = self.bert_sc(**batch)
        labels_predicted = output.logits.argmax(-1)
        num_correct = (labels_predicted == labels).sum().item()
        accuracy = num_correct / labels.size(0)
        self.log("accuracy", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

checkpoint = pl.callbacks.ModelCheckpoint(
    monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True, dirpath="model/"
)

trainer = pl.Trainer(gpus=1, max_epochs=10, callbacks=[checkpoint])
/home/taka/.local/share/virtualenvs/nlp_with_bert-KUEW-OT2/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: Checkpoint directory model/ exists and is not empty.
  warnings.warn(*args, **kwargs)


GPU available: True, used: True
TPU available: False, using: 0 TPU cores
model = BertForSequenceClassification_pl(MODEL_NAME, num_labels=9, lr=1e-5)

trainer.fit(model, dataloader_train, dataloader_val)

Some weights of the model checkpoint at cl-tohoku/bert-base-japanese-whole-word-masking were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.decoder.weight', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at cl-tohoku/bert-base-japanese-whole-word-masking and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name    | Type                          | Params
----------------------------------------------------------
0 | bert_sc | BertForSequenceClassification | 110 M
----------------------------------------------------------
110 M     Trainable params
0         Non-trainable params
110 M     Total params
442.497   Total estimated model params size (MB)
/home/taka/.local/share/virtualenvs/nlp_with_bert-KUEW-OT2/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, val dataloader 0, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  warnings.warn(*args, **kwargs)

/home/taka/.local/share/virtualenvs/nlp_with_bert-KUEW-OT2/lib/python3.9/site-packages/pytorch_lightning/utilities/distributed.py:52: UserWarning: The dataloader, train dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 8 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  warnings.warn(*args, **kwargs)
Epoch 9: 100%|██████████| 129/129 [01:16<00:00,  1.69it/s, loss=0.013, v_num=1]

"""
