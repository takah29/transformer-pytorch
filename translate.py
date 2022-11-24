import sys
from pathlib import Path

import torch

from libs.text_pair_dataset import get_vocab
from libs.text_encoder import TextEncoder, create_tokenizer
from libs.transformer import Transformer
from libs.transformer_predictor import TransformerPredictor


def create_instance(model_path: Path, enc_word_freqs_path: Path, dec_word_freqs_path: Path, device):
    enc_vocab = get_vocab(enc_word_freqs_path)
    dec_vocab = get_vocab(dec_word_freqs_path)
    transformer = Transformer.create(len(enc_vocab), len(dec_vocab))
    transformer.load_state_dict(torch.load(model_path))
    transformer.to(device)

    predictor = TransformerPredictor(transformer, dec_vocab["<bos>"], dec_vocab["<eos>"])

    enc_tokenizer = create_tokenizer(lang="en")
    enc_text_encoder = TextEncoder(enc_tokenizer, enc_vocab)

    dec_tokenizer = create_tokenizer(lang="ja")
    dec_text_encoder = TextEncoder(dec_tokenizer, dec_vocab)

    return predictor, enc_text_encoder, dec_text_encoder


def translate(text: str, predictor, enc_text_encoder, bos_id, eos_id, dec_text_encoder, device):
    encoded_text = enc_text_encoder.encode(text)
    input_text = torch.tensor(
        [bos_id] + encoded_text + [eos_id],
        dtype=torch.long,
    ).to(device)

    input_mask = torch.zeros(input_text.shape, dtype=torch.bool).to(device)

    output = predictor.predict(input_text, input_mask)
    translated_text = dec_text_encoder.decode(list(output)[1:-1], sep=" ")

    return translated_text


def main():
    base_path = Path(__file__).resolve().parent

    # モデルファイルパス
    model_dir_path = base_path / "models"
    if (model_dir_path / "model.pth").exists():
        model_path = model_dir_path / "model.pth"
    else:
        model_path = model_dir_path.glob("*.pth")[-1]

    # 単語頻度ファイルのパス
    word_freqs_dir = base_path / "word_freqs"
    enc_word_freqs_path = word_freqs_dir / "enc_word_freqs.json"
    dec_word_freqs_path = word_freqs_dir / "dec_word_freqs.json"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor, enc_text_encoder, dec_text_encoder = create_instance(
        model_path, enc_word_freqs_path, dec_word_freqs_path, device
    )
    bos_id = enc_text_encoder.get_bos_id()
    eos_id = enc_text_encoder.get_eos_id()

    while True:
        print("text: ", end="")
        text = input().strip()
        translated_text = translate(
            text, predictor, enc_text_encoder, bos_id, eos_id, dec_text_encoder, device
        )
        print(text)
        print(f"   -> {translated_text}")


if __name__ == "__main__":
    main()
