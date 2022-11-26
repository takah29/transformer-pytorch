import torch


class TransformerGreedyPredictor:
    def __init__(self, transformer, bos_id: int, eos_id: int, maxlen: int = 100):
        self._transformer = transformer
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._maxlen = maxlen

    @torch.inference_mode()
    def predict(self, enc_x: torch.Tensor) -> torch.Tensor:
        """Greedy Decoding"""

        self._transformer.eval()

        bos_id = torch.LongTensor([self._bos_id]).to(enc_x)
        eos_id = torch.LongTensor([self._eos_id]).to(enc_x)
        enc_x = torch.cat([bos_id, enc_x, eos_id])
        enc_mask = torch.zeros(enc_x.shape, dtype=torch.bool).to(enc_x)

        # (seq_size, ) -> (1, seq_size)
        enc_x.unsqueeze_(0)
        enc_mask.unsqueeze_(0)

        # (1, seq_size) -> (1, seq_size, n_dim)
        enc_y = self._transformer.encoder(enc_x, enc_mask)

        # -> (1, 1)
        dec_x = (torch.ones(1, 1, dtype=torch.long) * self._bos_id).to(enc_x.device)
        dec_mask = torch.zeros(dec_x.shape, dtype=torch.bool).to(enc_x.device)

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

            # <eos>の場合はmaxlenまで繰り返さずに処理を終了する
            if next_word_id.item() == self._eos_id:
                break

        return dec_x[0][1:-1]  # <bos>と<eos>を除く


class TransformerBeamSearchPredictor:
    def __init__(
        self,
        transformer,
        bos_id: int,
        eos_id: int,
        maxlen: int = 100,
        beam_size: int = 1000,
        alpha: float = 1.6,
    ):
        self._transformer = transformer
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._maxlen = maxlen
        self._k = beam_size
        self._alpha = alpha  # alpha > 0 なら出力が長くなり、alpha < 0 なら出力が短くなる

    @torch.inference_mode()
    def predict(self, enc_x: torch.Tensor) -> torch.Tensor:
        """Beam Search Decoding"""

        self._transformer.eval()

        bos_id = torch.LongTensor([self._bos_id]).to(enc_x)
        eos_id = torch.LongTensor([self._eos_id]).to(enc_x)
        enc_x = torch.cat([bos_id, enc_x, eos_id])
        enc_mask = torch.zeros(enc_x.shape, dtype=torch.bool).to(enc_x)

        # (seq_size, ) -> (1, seq_size)
        enc_x.unsqueeze_(0)
        enc_mask.unsqueeze_(0)

        # (1, seq_size) -> (1, seq_size, n_dim)
        enc_y = self._transformer.encoder(enc_x, enc_mask)

        # -> (1, 1)
        dec_x = (torch.ones(1, 1, dtype=torch.long) * self._bos_id).to(enc_x.device)
        dec_mask = torch.zeros(dec_x.shape, dtype=torch.bool).to(enc_x.device)

        # (1,)
        scores = torch.Tensor([0.0]).to(dec_x.device)

        for i in range(self._maxlen):
            # i = 0: dec_x (1, i + 1) -> (1, i + 1, vocab_size)
            # i > 0: dec_x (k, i + 1) -> (k, i + 1, vocab_size)
            dec_y = self._transformer.decoder(dec_x, enc_y, dec_mask, enc_mask)

            # i = 0 : (1, i + 1, vocab_size) -> (1, vocab_size)
            # i > 0 : (k, i + 1, vocab_size) -> (k, vocab_size)
            log_probs = torch.log_softmax(dec_y[:, -1], dim=1)

            # 系列長が長い場合は確率を低くする
            log_probs = log_probs / self.sequence_length_penalty(i + 1)

            # <eos>を対数尤度の計算に影響しないようにする
            log_probs[dec_x[:, -1] == self._eos_id, :] = 0

            # 対数尤度を計算する
            # i = 0: (1, 1) + (1, vocab_size) -> (1, vocab_size)
            # i > 0: (1, vocab_size) + (k, vocab_size) -> (k, vocab_size)
            scores = scores.unsqueeze(1) + log_probs

            # i = 0: (1, vocab_size) -> (k, )
            # i > 0: (k, vocab_size) -> (k, )
            scores, indices = torch.topk(scores.reshape(-1), self._k)

            # 選択したk要素の親ノード番号(0 - k-1)を算出
            vocab_size = dec_y.shape[-1]
            beam_indices = torch.divide(indices, vocab_size, rounding_mode="floor")

            # 選択したk要素の単語ID
            token_indices = torch.remainder(indices, vocab_size)

            next_decoder_input_list = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                # i = 0: (1, i + 1) -> (i + 1, )
                # i > 0: (k, i + 1) -> (i + 1, )
                prev_decoder_input = dec_x[beam_index]

                # 前回のデコーダーの入力の最後が<eos>の場合、それ以降の出力をすべて<eos>に置き換える
                if prev_decoder_input[-1] == self._eos_id:
                    token_index = self._eos_id

                # -> (1, )
                token_index = torch.LongTensor([token_index]).to(enc_x)

                # (i + 1, ), (1, ) -> (i + 2, )
                next_decoder_input = torch.cat([prev_decoder_input, token_index])

                next_decoder_input_list.append(next_decoder_input)

            # -> (k, i + 2)
            dec_x = torch.vstack(next_decoder_input_list)

            # k個すべての入力の最後が<eos>の場合、探索を終了する
            if (dec_x[:, -1] == self._eos_id).sum() == self._k:
                break

            # i > 0 ではバッチサイズkとして処理するので、合わせてエンコーダの出力サイズを拡張する
            if i == 0:
                # (1, seq_size, n_dim) -> (k, seq_size, n_dim)
                enc_y = enc_y.expand(self._k, *enc_y.shape[1:])

        # 対数尤度が最大の出力を取得する
        decoder_output, _ = max(zip(dec_x, scores), key=lambda x: x[1])

        # <bos>と<eos>を除く
        decoder_output = decoder_output[decoder_output != self._eos_id][1:]

        return decoder_output

    def sequence_length_penalty(self, length: int) -> float:
        return ((5 + length) / (5 + 1)) ** self._alpha
