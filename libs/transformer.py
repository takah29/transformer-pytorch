import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, n_dim, head_num, masking=False, dropout_rate=0.1):
        # 実装の簡易化のため次元をヘッド数で割り切れるかチェックする
        if n_dim % head_num != 0:
            raise ValueError("n_dim % head_num is not 0.")

        super().__init__()

        self.n_dim = n_dim
        self.head_num = head_num
        self.masking = masking

        self.head_dim = n_dim // head_num
        self.sqrt_head_dim = self.head_dim**0.5

        self.q_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.k_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.v_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_linear = nn.Linear(n_dim, n_dim)

    def forward(self, x1, x2, x2_mask=None):
        batch_size, x1_seq_size, _ = x1.size()  # seq_sizeはtoken_sizeと同じ
        _, x2_seq_size, _ = x2.size()

        # リニア層をヘッドごとに用意する方法もあるが、リニア層を適用したあと分割する
        # (batch, seq_size, n_dim) -> (batch, seq_size, n_dim)
        q = self.q_linear(x1)
        k = self.k_linear(x2)
        v = self.v_linear(x2)

        # (batch, seq_size, n_dim) -> (batch, seq_size, head_num, head_dim)
        q = q.view(batch_size, x1_seq_size, self.head_num, self.head_dim)
        k = k.view(batch_size, x2_seq_size, self.head_num, self.head_dim)
        v = v.view(batch_size, x2_seq_size, self.head_num, self.head_dim)

        # (batch, seq_size, head_num, head_dim) -> (batch, head_num, seq_size, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # (batch, head_num, x2_seq_size, head_dim) -> (batch, head_num, head_dim, x2_seq_size)
        k_transpose = k.transpose(2, 3)

        # (batch, head_num, x1_seq_size, head_dim) @ (batch, head_num, head_dim, x2_seq_size)
        # -> (batch, head_num, x1_seq_size, x2_seq_size)
        dots = (q @ k_transpose) / self.sqrt_head_dim

        # パディング部分にマスクを適用する
        if x2_mask is not None:
            # (batch, x2_seq_size) -> (batch, 1, 1, x2_seq_size)
            x2_mask = x2_mask.unsqueeze(-2).unsqueeze(-2)

            # (batch, head_num, x1_seq_size, x2_seq_size) * (batch, 1, 1, x2_seq_size)
            # -> (batch, head_num, x1_seq_size, x2_seq_size)
            dots = dots * x2_mask

        # 後続情報にマスクを適用する
        if self.masking:
            assert x1_seq_size == x2_seq_size  # self-attentionを仮定しているのでサイズが同じでないといけない

            # (batch, head_num, x1_seq_size, x1_seq_size) * (1, x1_seq_size, x1_seq_size)
            # -> (batch, head_num, x1_seq_size, x1_seq_size)
            dots = dots * MultiheadAttention._subsequent_mask(x1_seq_size).to(dots.device)

        # (batch, head_num, x1_seq_size, x2_seq_size) @ (batch, head_num, x2_seq_size, head_dim)
        # -> (batch, head_num, x1_seq_size, head_dim)
        attention_weight = self.dropout(F.softmax(dots, dim=-1))
        attention = attention_weight @ v

        # (batch, head_num, x1_seq_size, head_dim) -> (batch, x1_seq_size, n_dim)
        y = attention.transpose(1, 2).reshape(batch_size, x1_seq_size, self.n_dim)

        y = self.out_linear(y)

        return y

    @staticmethod
    def _subsequent_mask(size):
        attention_shape = (1, size, size)
        # 三角行列の生成
        subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).to(torch.uint8)

        return subsequent_mask == 0  # 1と0を反転


class FeedForwardNetwork(nn.Module):
    def __init__(self, n_dim, hidden_dim, dropout_rate=0.1):
        super().__init__()

        self.linear1 = nn.Linear(n_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        y = torch.relu(self.linear1(x))
        y = self.dropout(y)
        y = self.linear2(y)

        return y


class PositionalEncoder(nn.Module):
    def __init__(self, n_dim, dropout_rate=0.1, maxlen=1000):
        super().__init__()

        self.n_dim = n_dim
        self.maxlen = maxlen

        self.dropout = nn.Dropout(dropout_rate)
        self.register_buffer("embedding_pos", self._calc_pe(maxlen))

    def _calc_pe(self, maxlen):
        result = []
        pos_v = torch.arange(maxlen)

        for i in range(self.n_dim):
            if i % 2 == 0:
                v = torch.sin(pos_v / 10000 ** (i / self.n_dim))
            elif i % 2 == 1:
                v = torch.cos(pos_v / 10000 ** (i / self.n_dim))
            result.append(v)

        return torch.vstack(result).transpose(1, 0)

    def forward(self, x):
        return self.dropout(x + self.embedding_pos[: x.size(1), :])


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()

        self.attention = MultiheadAttention(n_dim, head_num)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm((n_dim))
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm((n_dim))

    def forward(self, x, x_mask=None):
        y = x + self.dropout1(self.attention(x, x, x_mask))
        y = self.norm1(y)
        y = y + self.dropout2(self.feedforward(y))
        y = self.norm2(y)

        return y


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_dim, hidden_dim, n_enc_blocks, head_num, dropout_rate=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.positional_encoder = PositionalEncoder(n_dim, dropout_rate)
        self.enc_blocks = nn.ModuleList(
            [TransformerEncoderBlock(n_dim, hidden_dim, head_num) for _ in range(n_enc_blocks)]
        )

    def forward(self, x, src_mask=None):
        y = self.embedding(x)
        y = self.positional_encoder(y)

        for enc_block in self.enc_blocks:
            y = enc_block(y, src_mask)

        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_dim, hidden_dim, head_num, dropout_rate=0.1):
        super().__init__()

        self.masked_attention = MultiheadAttention(n_dim, head_num, masking=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm((n_dim))
        self.attention = MultiheadAttention(n_dim, head_num)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm((n_dim))
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm((n_dim))

    def forward(self, x, z, x_mask=None, z_mask=None):
        y = x + self.dropout1(self.masked_attention(x, x, x_mask))
        y = self.norm1(y)
        y = y + self.dropout2(self.attention(y, z, z_mask))
        y = self.norm2(y)
        y = y + self.dropout3(self.feedforward(y))
        y = self.norm3(y)

        return y


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, n_dim, hidden_dim, n_dec_blocks, head_num, dropout_rate=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.pe = PositionalEncoder(n_dim, dropout_rate)
        self.dec_blocks = nn.ModuleList(
            [TransformerDecoderBlock(n_dim, hidden_dim, head_num) for _ in range(n_dec_blocks)]
        )
        self.out_linear = nn.Linear(n_dim, vocab_size)

    def forward(self, x, z, x_mask=None, z_mask=None):
        y = self.embedding(x)
        y = self.pe(y)

        for dec_block in self.dec_blocks:
            y = dec_block(y, z, x_mask, z_mask)

        y = self.out_linear(y)
        return y


class TransformerClassifier(nn.Module):
    """TransformerEncoderを使用した分類用ネットワーク"""

    def __init__(
        self, n_classes, vocab_size, n_dim, hidden_dim, n_enc_blocks, head_num, dropout_rate=0.1
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, n_dim, hidden_dim, n_enc_blocks, head_num, dropout_rate
        )
        self.linear = nn.Linear(n_dim, n_classes)

    def forward(self, x, mask=None):
        y = self.encoder(x, mask)
        y = torch.mean(y, dim=1)
        y = self.linear(y)

        return y


class Transformer(nn.Module):
    def __init__(
        self,
        enc_vocab_size,
        dec_vocab_size,
        n_dim,
        hidden_dim,
        n_enc_blocks,
        n_dec_blocks,
        head_num,
        dropout_rate=0.1,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            enc_vocab_size, n_dim, hidden_dim, n_enc_blocks, head_num, dropout_rate
        )
        self.decoder = TransformerDecoder(
            dec_vocab_size, n_dim, hidden_dim, n_dec_blocks, head_num, dropout_rate
        )

        self._initialize()

    def _initialize(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, enc_x, dec_x, enc_mask, dec_mask):
        y = self.encoder(enc_x, enc_mask)
        y = self.decoder(dec_x, y, dec_mask, enc_mask)

        return y


if __name__ == "__main__":
    from pathlib import Path
    from dataset import TextPairDataset
    from torch.utils.data import DataLoader

    # 事前にbuild_word_freqs.pyを実行してデータセットのダウンロードと頻度辞書の作成を行っておく
    en_txt_file_path = Path("dataset/small_parallel_enja-master/dev.en").resolve()
    ja_txt_file_path = Path("dataset/small_parallel_enja-master/dev.ja").resolve()
    en_word_freqs_path = Path("word_freqs_en.json").resolve()
    ja_word_freqs_path = Path("word_freqs_ja.json").resolve()

    text_pair_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )
    data_loader = DataLoader(list(text_pair_dataset)[:10], batch_size=2, shuffle=True)

    enc_vocab_size, dec_vocab_size = text_pair_dataset.get_vocab_size()
    params = {
        "enc_vocab_size": enc_vocab_size,
        "dec_vocab_size": dec_vocab_size,
        "n_dim": 64,
        "hidden_dim": 32,
        "n_enc_blocks": 1,
        "n_dec_blocks": 1,
        "head_num": 1,
        "dropout_rate": 0.1,
    }
    transformer = Transformer(**params)

    for i, batch in enumerate(data_loader):
        print("iter:", i + 1)
        print("input shape:", batch["enc_input"]["text"].size())
        y = transformer(
            batch["enc_input"]["text"],
            batch["dec_input"]["text"],
            batch["enc_input"]["mask"],
            batch["dec_input"]["mask"],
        )
        print("output shape:", y.shape)
