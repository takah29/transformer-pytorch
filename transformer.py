from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

n_batch = 2
token_size = 256
n_dim = 768

x = torch.rand(n_batch, token_size, n_dim)

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


# %%
attention = Attention(n_dim, 200)
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


ffn = FeedForwardNetwork(n_dim, 200)
y = ffn.forward(x)
print(y.shape)


class PositionalEncoder(nn.Module):
    def __init__(self, n_dim):
        self.n_dim = n_dim

    def eval_pe(self, x):
        _, token_size, self.n_dim = x.shape
        result = []
        pos_v = torch.arange(token_size)
        for i in range(self.n_dim):
            if i % 2 == 0:
                v = torch.sin(pos_v / 10000 ** (i / self.n_dim))
            elif i % 2 == 1:
                v = torch.cos(pos_v / 10000 ** (i / self.n_dim))
            result.append(v)

        # print(torch.vstack(result).transpose(1, 0)[0])
        # print(torch.vstack(result).transpose(1, 0)[1])

        return torch.vstack(result).transpose(1, 0)

    def forward(self, x):
        return x + self.eval_pe(x)


pe = PositionalEncoder(n_dim)
y = pe.forward(torch.zeros(2, token_size, n_dim))
print(y[0][1, 0], y[1][1, 0])

# %%
class TrasformerEncoder(nn.Module):
    def init(self, input_dim, output_dim):
        self.embedding = "dummy"
        self.skdfhfsjhdfkh
        self.asdakpdkpaksp
