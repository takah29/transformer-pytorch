import torch


class TransformerPredictor:
    def __init__(self, transformer, bos_id, eos_id, maxlen=100):
        self._transformer = transformer
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._maxlen = maxlen

    def predict(self, enc_x, enc_mask):
        # enc_x shape = (seq_size, )

        self._transformer.eval()
        with torch.inference_mode():
            # (seq_size, ) -> (1, seq_size)
            enc_x.unsqueeze_(0)
            enc_mask.unsqueeze_(0)

            # (1, seq_size) -> (1, seq_size, n_dim)
            enc_y = self._transformer.encoder(enc_x, enc_mask)

            # -> (1, 1)
            dec_x = (torch.ones(1, 1, dtype=torch.long) * self._bos_id).to(enc_x.device)
            dec_mask = torch.zeros(dec_x.shape, dtype=torch.bool).to(enc_x.device)

            # print(dec_x, dec_mask)
            for _ in range(self._maxlen):
                # (1, i + 1) -> (1, i + 1, vocab_size)
                dec_y = self._transformer.decoder(dec_x, enc_y, dec_mask, enc_mask)

                # (1, i + 1, vocab_size) -> (1,)
                next_word_id = dec_y[:, -1].argmax(dim=1)

                # -> (1, 1)
                next_word_id.unsqueeze_(0)

                # (1, i + 1) , (1, 1) -> (1, i + 2)
                dec_x = torch.cat((dec_x, next_word_id), dim=1)
                dec_mask = torch.zeros(dec_x.shape, dtype=torch.bool).to(enc_x.device)

                # <eos>の場合はmaxlenまで繰り返さずに処理を打ち切る
                if next_word_id.item() == self._eos_id:
                    break

        return dec_x[0]

    def forward(self, enc_x, enc_mask):
        self._transformer.eval()
        with torch.inference_mode():
            # (seq_size, ) -> (1, seq_size)
            enc_x.unsqueeze_(0)
            enc_mask.unsqueeze_(0)
            dec_x = (torch.ones(1, 1, dtype=torch.long) * self._bos_id).to(enc_x.device)
            dec_mask = torch.ones(dec_x.shape, dtype=torch.uint8).to(enc_x.device)

            y = self._transformer.forward(enc_x, dec_x, enc_mask, dec_mask)
        return y[:, -1].argmax(dim=1)
