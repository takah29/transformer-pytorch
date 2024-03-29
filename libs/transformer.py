import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, n_dim: int, head_num: int, dropout_rate: float = 0.1):
        # 実装の簡易化のため次元をヘッド数で割り切れるかチェックする
        if n_dim % head_num != 0:
            raise ValueError("n_dim % head_num is not 0.")

        super().__init__()

        self.n_dim = n_dim
        self.head_num = head_num

        self.head_dim = n_dim // head_num
        self.sqrt_head_dim = self.head_dim**0.5

        self.q_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.k_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.v_linear = nn.Linear(n_dim, n_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.out_linear = nn.Linear(n_dim, n_dim)

    def forward(
        self, x1: Tensor, x2: Tensor, x2_mask: Tensor | None = None, attn_mask: Tensor | None = None
    ) -> Tensor:
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
            dots *= torch.logical_not(x2_mask)

        # 後続情報にマスクを適用する
        if attn_mask is not None:
            assert x1_seq_size == x2_seq_size  # self-attentionを仮定しているのでサイズが同じでないといけない
            dots[:, :, attn_mask] = float("-inf")

        attention_weight = self.dropout(F.softmax(dots, dim=-1))

        # (batch, head_num, x1_seq_size, x2_seq_size) @ (batch, head_num, x2_seq_size, head_dim)
        # -> (batch, head_num, x1_seq_size, head_dim)
        attention = attention_weight @ v

        # (batch, head_num, x1_seq_size, head_dim) -> (batch, x1_seq_size, n_dim)
        y = attention.transpose(1, 2).reshape(batch_size, x1_seq_size, self.n_dim)

        y = self.out_linear(y)

        return y

    @staticmethod
    def _subsequent_mask(size: int) -> Tensor:
        attention_shape = (size, size)
        # 三角行列の生成
        subsequent_mask = torch.triu(torch.ones(attention_shape), diagonal=1).to(torch.bool)

        return subsequent_mask


class FeedForwardNetwork(nn.Module):
    def __init__(self, n_dim: int, hidden_dim: int, dropout_rate: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(n_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        y = torch.relu(self.linear1(x))
        y = self.dropout(y)
        y = self.linear2(y)

        return y


class PositionalEncoder(nn.Module):
    def __init__(self, n_dim: int, dropout_rate: float = 0.1, maxlen: int = 1000):
        super().__init__()

        self.n_dim = n_dim
        self.maxlen = maxlen

        self.dropout = nn.Dropout(dropout_rate)
        emb_pos: Tensor = self._calc_pe(maxlen)
        self.register_buffer("embedding_pos", emb_pos)

    def _calc_pe(self, maxlen: int) -> Tensor:
        result = []
        pos_v = torch.arange(maxlen)

        for i in range(self.n_dim):
            if i % 2 == 0:
                v = torch.sin(pos_v / 10000 ** (i / self.n_dim))
            elif i % 2 == 1:
                v = torch.cos(pos_v / 10000 ** (i / self.n_dim))
            result.append(v)

        return torch.vstack(result).transpose(1, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.embedding_pos[: x.size(1), :])


class TransformerEncoderBlock(nn.Module):
    def __init__(self, n_dim: int, hidden_dim: int, head_num: int, dropout_rate: float = 0.1):
        super().__init__()

        self.attention = MultiheadAttention(n_dim, head_num, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm((n_dim))
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm((n_dim))

    def forward(self, x: Tensor, x_mask: Tensor | None = None) -> Tensor:
        y = x + self.dropout1(self.attention(x, x, x_mask))
        y = self.norm1(y)
        y = y + self.dropout2(self.feedforward(y))
        y = self.norm2(y)

        return y


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_dim: int,
        hidden_dim: int,
        n_enc_blocks: int,
        head_num: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.positional_encoder = PositionalEncoder(n_dim, dropout_rate)
        self.enc_blocks = nn.ModuleList(
            [TransformerEncoderBlock(n_dim, hidden_dim, head_num) for _ in range(n_enc_blocks)]
        )

        self.sqrt_embedding_size = n_dim**0.5

    def forward(self, x: Tensor, src_mask: Tensor | None = None) -> Tensor:
        y = self.embedding(x) * self.sqrt_embedding_size
        y = self.positional_encoder(y)

        for enc_block in self.enc_blocks:
            y = enc_block(y, src_mask)

        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_dim: int, hidden_dim: int, head_num: int, dropout_rate: float = 0.1):
        super().__init__()

        self.masked_attention = MultiheadAttention(n_dim, head_num, dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm((n_dim))
        self.attention = MultiheadAttention(n_dim, head_num, dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm((n_dim))
        self.feedforward = FeedForwardNetwork(n_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.norm3 = nn.LayerNorm((n_dim))

    def forward(
        self, x: Tensor, z: Tensor, x_mask: Tensor | None = None, z_mask: Tensor | None = None
    ) -> Tensor:
        attn_mask = MultiheadAttention._subsequent_mask(x.shape[1])
        attn_mask = attn_mask.to(x.device)
        y = x + self.dropout1(self.masked_attention(x, x, x2_mask=x_mask, attn_mask=attn_mask))
        y = self.norm1(y)
        y = y + self.dropout2(self.attention(y, z, z_mask))
        y = self.norm2(y)
        y = y + self.dropout3(self.feedforward(y))
        y = self.norm3(y)

        return y


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_dim: int,
        hidden_dim: int,
        n_dec_blocks: int,
        head_num: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.pe = PositionalEncoder(n_dim, dropout_rate)
        self.dec_blocks = nn.ModuleList(
            [TransformerDecoderBlock(n_dim, hidden_dim, head_num) for _ in range(n_dec_blocks)]
        )
        self.out_linear = nn.Linear(n_dim, vocab_size)

        self.sqrt_embedding_size = n_dim**0.5

    def forward(
        self, x: Tensor, z: Tensor, x_mask: Tensor | None = None, z_mask: Tensor | None = None
    ) -> Tensor:
        y = self.embedding(x) * self.sqrt_embedding_size
        y = self.pe(y)

        for dec_block in self.dec_blocks:
            y = dec_block(y, z, x_mask, z_mask)

        y = self.out_linear(y)
        return y


class TransformerClassifier(nn.Module):
    """TransformerEncoderを使用した分類用ネットワーク"""

    def __init__(
        self,
        n_classes: int,
        vocab_size: int,
        n_dim: int,
        hidden_dim: int,
        n_enc_blocks: int,
        head_num: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.encoder = TransformerEncoder(
            vocab_size, n_dim, hidden_dim, n_enc_blocks, head_num, dropout_rate
        )
        self.linear = nn.Linear(n_dim, n_classes)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        y = self.encoder(x, mask)
        y = torch.mean(y, dim=1)
        y = self.linear(y)

        return y


class Transformer(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int,
        dec_vocab_size: int,
        n_dim: int,
        hidden_dim: int,
        n_enc_blocks: int,
        n_dec_blocks: int,
        head_num: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.n_dim = n_dim
        self.hidden_dim = hidden_dim
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.head_num = head_num
        self.dropout_rate = dropout_rate

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

    def forward(self, enc_x: Tensor, dec_x: Tensor, enc_mask: Tensor, dec_mask: Tensor):
        enc_y = self.encoder(enc_x, enc_mask)
        y = self.decoder(dec_x, enc_y, dec_mask, enc_mask)

        return y

    @staticmethod
    def create(enc_vocab_size: int, dec_vocab_size: int):
        params = {
            "enc_vocab_size": enc_vocab_size,
            "dec_vocab_size": dec_vocab_size,
            "n_dim": 128,
            "hidden_dim": 500,
            "n_enc_blocks": 2,
            "n_dec_blocks": 2,
            "head_num": 8,
            "dropout_rate": 0.3,
        }
        return Transformer(**params)


if __name__ == "__main__":
    from pathlib import Path

    from torch.utils.data import DataLoader

    from text_pair_dataset import TextPairDataset

    base_path = Path(__file__).resolve().parents[1] / "multi30k_dataset"
    en_txt_file_path = base_path / "src_train_texts.txt"
    ja_txt_file_path = base_path / "tgt_train_texts.txt"
    en_word_freqs_path = base_path / "src_word_freqs.json"
    ja_word_freqs_path = base_path / "tgt_word_freqs.json"

    text_pair_dataset = TextPairDataset.create(
        en_txt_file_path, ja_txt_file_path, en_word_freqs_path, ja_word_freqs_path
    )
    data_loader = DataLoader(list(text_pair_dataset)[:10], batch_size=2, shuffle=True)

    enc_vocab_size, dec_vocab_size = text_pair_dataset.get_vocab_size()
    transformer = Transformer.create(enc_vocab_size, dec_vocab_size)

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
