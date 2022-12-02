from pathlib import Path
import json

import torch

from libs.text_pair_dataset import get_vocab
from libs.text_encoder import TextEncoder, create_tokenizer
from libs.transformer import Transformer
from libs.transformer_predictor import TransformerBeamSearchPredictor


def create_instance(
    params: dict,
    model_path: Path,
    src_word_freqs_path: Path,
    tgt_word_freqs_path: Path,
    src_min_freq: int,
    tgt_min_freq: int,
    device,
):
    src_vocab = get_vocab(src_word_freqs_path, src_min_freq)
    tgt_vocab = get_vocab(tgt_word_freqs_path, tgt_min_freq)

    params["enc_vocab_size"] = len(src_vocab)
    params["dec_vocab_size"] = len(tgt_vocab)

    transformer = Transformer(**params)
    transformer.load_state_dict(torch.load(model_path))
    transformer.to(device)

    predictor = TransformerBeamSearchPredictor(
        transformer, tgt_vocab["<bos>"], tgt_vocab["<eos>"], maxlen=100
    )

    src_tokenizer = create_tokenizer(lang="en")
    src_text_encoder = TextEncoder(src_tokenizer, src_vocab)

    src_tokenizer = create_tokenizer(lang="ja")
    tgt_text_encoder = TextEncoder(src_tokenizer, tgt_vocab)

    return predictor, src_text_encoder, tgt_text_encoder


def translate(text: str, predictor, enc_text_encoder, dec_text_encoder, device):
    encoded_text = enc_text_encoder.encode(text)
    input_text = torch.tensor(encoded_text, dtype=torch.long).to(device)

    output = predictor.predict(input_text)

    translated_text = dec_text_encoder.decode(list(output), sep=" ")

    return translated_text


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Translate with pre-trained odel..")
    parser.add_argument(
        "dataset_dir", help="Dataset root directory with pre-trained model.", type=str
    )

    args = parser.parse_args()

    base_path = Path(args.dataset_dir).resolve()

    # モデルファイルパス
    model_dir_path = base_path / "models"

    if (model_dir_path / "model.pth").exists():
        model_path = model_dir_path / "model.pth"
    else:
        model_path = model_dir_path.glob("*.pth")[-1]

    # 単語頻度ファイルのパス
    src_word_freqs_path = base_path / "src_word_freqs.json"
    tgt_word_freqs_path = base_path / "tgt_word_freqs.json"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # パラメータ設定の読み込み
    with (base_path / "settings.json").open("r") as f:
        settings = json.load(f)

    params = settings["params"]
    src_min_freq = settings["min_freq"]["source"]
    tgt_min_freq = settings["min_freq"]["target"]

    predictor, src_text_encoder, tgt_text_encoder = create_instance(
        params,
        model_path,
        src_word_freqs_path,
        tgt_word_freqs_path,
        src_min_freq,
        tgt_min_freq,
        device,
    )

    while True:
        text = input("text: ").strip()

        if text == "":
            continue

        translated_text = translate(text, predictor, src_text_encoder, tgt_text_encoder, device)
        print(text)
        print(f"   -> {translated_text}")


if __name__ == "__main__":
    main()
