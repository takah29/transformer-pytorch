from pathlib import Path
import numpy as np
import torch
from torch import layer_norm, nn
from torch.nn import functional as F

n_batch = 2
token_size = 256
input_dim = 768

vocab_size = 1000

x = torch.rand(n_batch, token_size, input_dim)


# %%
class Attention(nn.Module):
    def __init__(self, n_dim, hidden_dim):
        super().__init__()
        self.q_linear = nn.Linear(n_dim, hidden_dim)
        self.k_linear = nn.Linear(n_dim, hidden_dim)
        self.v_linear = nn.Linear(n_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, n_dim)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        y = torch.matmul(F.softmax(torch.matmul(q, k.transpose(1, 2)), dim=2), v)
        y = self.out_linear(y)

        return y


class MultiheadAttention(nn.Module):
    def __init__(self, ndim, hidden_dim, num):
        super().__init__()

    def forward(self):
        pass


# %%
attention = Attention(input_dim, 200)
y = attention.forward(x)
print(y.shape)


# %%
class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activate = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.activate(x)


ffn = FeedForwardNetwork(input_dim, 200)
y = ffn.forward(x)
print(y.shape)


class PositionalEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def eval_pe(self, x):
        _, token_size, self.input_dim = x.shape
        result = []
        pos_v = torch.arange(token_size)
        for i in range(self.input_dim):
            if i % 2 == 0:
                v = torch.sin(pos_v / 10000 ** (i / self.input_dim))
            elif i % 2 == 1:
                v = torch.cos(pos_v / 10000 ** (i / self.input_dim))
            result.append(v)

        # print(torch.vstack(result).transpose(1, 0)[0])
        # print(torch.vstack(result).transpose(1, 0)[1])

        return torch.vstack(result).transpose(1, 0)

    def forward(self, x):
        return x + self.eval_pe(x)


# %%
class TrasformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, 0)
        self.pe = PositionalEncoder(input_dim)
        self.attention = Attention(input_dim, hidden_dim)
        self.feedforward = FeedForwardNetwork(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm((token_size, input_dim))
        self.norm2 = nn.LayerNorm((token_size, input_dim))

    def forward(self, x):
        y = self.embedding(x)
        y = self.pe.forward(y)
        y = y + self.attention.forward(y)
        y = self.norm1(y)
        y = y + self.feedforward.forward(y)
        y = self.norm2(y)
        return y


class Test:
    # 分類用のネットワーク
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.enc = TrasformerEncoder(input_dim, hidden_dim)
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x):
        y = self.enc.forward(x)
        y = self.linear(x)
        return y


def main():
    pe = PositionalEncoder(input_dim)
    result = pe.forward(torch.rand((n_batch, token_size, input_dim)))
    print(result.size())

    enc = TrasformerEncoder(input_dim, 256)
    result = enc.forward(torch.randint(0, vocab_size, (n_batch, token_size)))
    print(result)


if __name__ == "__main__":
    main()
