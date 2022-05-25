# %%
from pathlib import Path
import janome
from janome.tokenizer import Tokenizer
import torchtext
import torch
from torch import nn
from torch.nn import functional as F
# import spacy
from torchtext.legacy import data
from torchtext.legacy import datasets
from torchtext.vocab import GloVe, FastText

tokenizer_ja = Tokenizer()

# %%
def tokenize_ja(text):
    return list(tokenizer_ja.tokenize(text, wakati=True))

print(tokenize_ja("私は人間です。"))


# %%
def tokenize_en(text):
    text = text.replace(".", " .")

    return text.split()

print(tokenize_en("I   am a human."))

# %%
TEXT_EN = data.Field(sequential=True, tokenize=tokenize_en, lower=True)
TEXT_JA = data.Field(sequential=True, tokenize=tokenize_ja, lower=True)

# %%
def make_parallel_dataset(data_en_path, data_ja_path, output_path):
    # すでにトークナイズされているがもとに戻す
    results = []
    with open(data_en_path, "r") as f_en, open(data_ja_path, "r") as f_ja:
        for line_en, line_ja in zip(f_en, f_ja):
            line_en = line_en.strip().replace(" .", ".").replace(" '", "'").replace(" ,", ",")
            line_ja = line_ja.strip().replace(" ", "")
            line = "\t".join((line_en, line_ja))
            results.append(line)

    with open(output_path, "w") as f:
        for line in results:
            f.write(f"{line}\n")


# %%
# make_parallel_dataset("./dev.en", "./dev.ja", "./dev.tsv")
# make_parallel_dataset("./test.en", "./test.ja", "./test.tsv")

# %%
train, test = data.TabularDataset.splits(
    path="./",
    train="dev.tsv",
    test="test.tsv",
    format="tsv",
    fields=[("text_en", TEXT_EN), ("text_ja", TEXT_JA)],
)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))


# %%
TEXT_EN.build_vocab(train, min_freq=1)
TEXT_JA.build_vocab(train, min_freq=1)

print(list(TEXT_EN.vocab.freqs.items())[:10])
print(list(TEXT_JA.vocab.freqs.items())[:10])
print(list(TEXT_EN.vocab.stoi.items())[:10])
print(list(TEXT_JA.vocab.stoi.items())[:10])


# %%
# 単語ベクトルデータがキャッシュにない場合はダウンロードする。英語6.6GB, 日本語1.4GBあるので時間がかかる
TEXT_EN.build_vocab(train, vectors=FastText(language="en"), min_freq=1)
TEXT_JA.build_vocab(train, vectors=FastText(language="ja"), min_freq=1)


# %%
train_iter, test_iter = data.Iterator.splits((train, test), batch_sizes=(2, 2), shuffle=True, device="cpu")
batch = next(iter(train_iter))
print(batch.text_en.shape)
print(batch.text_ja.shape)


# %%
class Attention(nn.Module):
    def init(self, n_features, out_features):
        super().init()
        self.q_linear = nn.Linear(n_features, out_features)
        self.k_linear = nn.Linear(n_features, out_features)
        self.v_linear = nn.Linear(n_features, out_features)
        self.out_linear = nn.Linear(out_features, n_features)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        y = torch.matmul(F.softmax(torch.matmul(q, k.transpose(1, 2)), dim=2), v)
        y = self.out_linear(y)

        return y


# %%
attention = Attention(768, 200)
y = attention.forward(torch.rand(2, 256, 768))
print(y.shape)

# %%
class FeedForwardNetwork(nn.Module):
    def init(self, input_dim, output_dim):
        super().init()

        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, input_dim)
        self.activate = nn.GELU()

    def forword(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.activate(x)

# %%
class TrasformerEncoder(nn.Module):
    def init(self, input_dim, output_dim):
        self.embedding = "dummy"
        self.skdfhfsjhdfkh
        self.asdakpdkpaksp


