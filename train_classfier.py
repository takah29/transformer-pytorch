import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformer import TransformerClassifier
from transformers import AutoTokenizer


def train(num_epoch, model, optimizer, train_x, train_t):
    criterion = nn.CrossEntropyLoss()
    # dataset = torch.utils.data.TensorDataset(train_x, train_t)
    data_loader = DataLoader(list(zip(train_x, train_t)), batch_size=100, shuffle=True)

    loss_list = []
    s = 0.0
    for epoch in range(num_epoch):
        print("epoch: ", epoch)
        s = 0.0
        for i, (x, t) in enumerate(data_loader):
            # print(i + 1, x.size(), t.size())
            y = model(x)
            # print(f"batch: {i+1}", x.size(), y, t)
            loss = criterion(y, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s += loss.item()
        loss_list.append(s)
        print(s)

    import matplotlib.pyplot as plt

    plt.plot(loss_list)
    plt.show()


def main():
    token_size = 256
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    token = tokenizer.encode_plus("お腹が痛いので遅れます。", padding="max_length", max_length=token_size)
    vocab_size = tokenizer.vocab_size

    print(vocab_size)
    print(token)

    exit()


def test():

    N = 1000
    vocab_size = 1000
    token_size = 256

    x1 = torch.randint(0, vocab_size // 2, (N, token_size))
    t1 = torch.zeros(N).to(torch.int64)
    x2 = torch.randint(vocab_size // 2, vocab_size, (N, token_size))
    t2 = torch.ones(N).to(torch.int64)

    X = torch.vstack((x1, x2))
    T = torch.hstack((t1, t2))

    param = {
        "n_classes": 2,
        "n_enc_blocks": 1,
        "vocab_size": vocab_size,
        "n_dim": 100,
        "hidden_dim": 16,
        "token_size": token_size,
        "head_num": 1,
    }

    model = TransformerClassifier(**param)

    optimizer = optim.Adam(model.parameters())

    num_epoch = 10
    train(num_epoch, model, optimizer, X, T)


if __name__ == "__main__":
    test()
