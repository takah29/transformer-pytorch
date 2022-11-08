import torch
from torch import nn
from torch.nn import functional as F


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
    def __init__(self, n_dim, hidden_dim, num):
        super().__init__()

    def forward(self):
        pass


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


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_dim, hidden_dim, token_size, n_blocks):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.pe = PositionalEncoder(n_dim)
        self.enc_blocks = [
            TransformerEncoderBlock(n_dim, hidden_dim, token_size) for _ in range(n_blocks)
        ]

    def forward(self, x):
        y = self.embedding(x)
        y = self.pe(y)

        for enc_block in self.enc_blocks:
            y = enc_block(y)

        return y


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim, hidden_dim, token_size):
        super().__init__()

        self.attention = Attention(n_dim, hidden_dim)
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.norm1 = nn.LayerNorm((token_size, n_dim))
        self.norm2 = nn.LayerNorm((token_size, n_dim))

    def forward(self, x):
        y = x + self.attention(x)
        y = self.norm1(y)
        y = y + self.feedforward(y)
        y = self.norm2(y)

        return y


class TransformerClassifier(nn.Module):
    # TransformerEncoderを使用した分類用ネットワーク
    def __init__(self, n_classes, vocab_size, n_dim, hidden_dim, token_size, n_blocks):
        super().__init__()

        self.encoder = TransformerEncoder(vocab_size, n_dim, hidden_dim, token_size, n_blocks)
        self.linear = nn.Linear(n_dim, n_classes)

    def forward(self, x):
        y = self.encoder(x)
        y = torch.mean(y, dim=1)
        y = self.linear(y)

        return y
