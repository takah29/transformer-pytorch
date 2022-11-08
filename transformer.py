import torch
from torch import nn
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.n_dim = n_dim

        self.q_linear = nn.Linear(n_dim, n_dim)
        self.k_linear = nn.Linear(n_dim, n_dim)
        self.v_linear = nn.Linear(n_dim, n_dim)
        # self.out_linear = nn.Linear(n_dim, n_dim)  # hidden_dim = n_dimとすればこのリニア層は必要ない

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        y = torch.matmul(
            F.softmax(torch.matmul(q, k.transpose(1, 2) / self.n_dim**0.5), dim=2), v
        )
        # y = self.out_linear(y)

        return y


class MultiheadAttention(nn.Module):
    def __init__(self, n_dim, head_num, mask=False):
        # 実装の簡易化のため次元をヘッド数で割り切れるかチェックする
        if n_dim % head_num != 0:
            raise ValueError("n_dim % head_num is not 0.")

        super().__init__()

        self.n_dim = n_dim
        self.head_num = head_num
        self.mask = mask

        self.head_dim = n_dim // head_num
        self.sqrt_head_dim = self.head_dim**0.5

        self.q_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.k_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.v_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.out_linear = nn.Linear(n_dim, n_dim)

    def forward(self, s, t, src_mask=None):
        batch_size, seq_size, _ = t.size()  # seq_sizeはtoken_sizeと同じ

        # リニア層をヘッドごとに用意する方法もあるが、リニア層を適用したあと分割する
        # (batch, seq_size, n_dim) -> (batch, seq_size, n_dim)
        q = self.q_linear(s)
        k = self.k_linear(t)
        v = self.v_linear(t)

        # (batch, seq_size, n_dim) -> (batch, seq_size, head_num, head_dim)
        q = q.view(batch_size, seq_size, self.head_num, self.head_dim)
        k = k.view(batch_size, seq_size, self.head_num, self.head_dim)
        v = v.view(batch_size, seq_size, self.head_num, self.head_dim)

        # (batch, seq_size, head_num, head_dim) -> (batch, head_num, seq_size, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch, head_num, head_dim, seq_size)
        k_transpose = k.transpose(2, 3)
        dots = (q @ k_transpose) / self.sqrt_head_dim

        # パディング部分などにマスクを適用する
        if src_mask is not None:
            src_mask = src_mask.unsqueeze(-2)
            dots = dots & src_mask

        # 後続情報にマスクを適用する
        if self.mask:
            dots = dots & MultiheadAttention._subsequent_mask(seq_size)

        attention = F.softmax(dots, dim=-1) @ v
        y = attention.transpose(1, 2).reshape(batch_size, seq_size, self.n_dim)
        y = self.out_linear(y)

        return y

    @staticmethod
    def _subsequent_mask(size):
        attn_shape = (1, size, size)  # [1, tgt_len, tgt_len]
        # 三角行列の生成
        subsequent_mask = torch.triu(torch.ones(attn_shape), k=1).astype(
            "uint8"
        )  # [1, tgt_len, tgt_len]
        # 1と0を反転する処理
        return subsequent_mask == 0


class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        # self.activate = nn.GELU()

    def forward(self, x):
        y = torch.relu(self.linear1(x))
        y = self.linear2(y)

        return y


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

        return torch.vstack(result).transpose(1, 0)

    def forward(self, x):
        return x + self.eval_pe(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim, hidden_dim, token_size, head_num):
        super().__init__()

        self.attention = MultiheadAttention(n_dim, head_num)
        self.norm1 = nn.LayerNorm((token_size, n_dim))
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.norm2 = nn.LayerNorm((token_size, n_dim))

    def forward(self, x):
        y = x + self.attention(x, x)
        y = self.norm1(y)
        y = y + self.feedforward(y)
        y = self.norm2(y)

        return y


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_dim, hidden_dim, token_size, n_blocks, head_num):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.pe = PositionalEncoder(n_dim)
        self.enc_blocks = [
            TransformerEncoderBlock(n_dim, hidden_dim, token_size, head_num)
            for _ in range(n_blocks)
        ]

    def forward(self, x):
        y = self.embedding(x)
        y = self.pe(y)

        for enc_block in self.enc_blocks:
            y = enc_block(y)

        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_dim, hidden_dim, token_size, head_num):
        super().__init__()

        self.attention = MultiheadAttention(n_dim, head_num)
        self.norm1 = nn.LayerNorm((token_size, n_dim))
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.norm2 = nn.LayerNorm((token_size, n_dim))

    def forward(self, x, z):
        y = x + self.attention(x, z)
        y = self.norm1(y)
        y = y + self.feedforward(y)
        y = self.norm2(y)

        return y


class TransformerClassifier(nn.Module):
    # TransformerEncoderを使用した分類用ネットワーク
    def __init__(self, n_classes, vocab_size, n_dim, hidden_dim, token_size, n_blocks, head_num):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, n_dim, hidden_dim, token_size, n_blocks, head_num
        )
        self.linear = nn.Linear(n_dim, n_classes)

    def forward(self, x):
        y = self.encoder(x)
        y = torch.mean(y, dim=1)
        y = self.linear(y)

        return y


"""
* AttentionのMaskingの実装について: https://qiita.com/FuwaraMiyasaki/items/239f3528053889847825#attention%E3%81%AEmasking%E3%81%AE%E5%AE%9F%E8%A3%85%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6
"""
