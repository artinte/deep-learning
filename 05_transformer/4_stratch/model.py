import torch
from positional_encoding import PositionalEncoding


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        d_model=256,
        nhead=8,
        n_ff=1024,
        nlayers=6,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.src_emb = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=n_ff,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = torch.nn.Linear(d_model, tgt_vocab_size)

    def make_src_mask(self, src):
        # This creates a boolean mask that is passed to src_key_padding_mask
        return src == self.src_pad_idx

    def make_tgt_mask(self, tgt):
        # Create the padding mask as a float tensor with a large negative value
        tgt_pad_mask = (tgt == self.tgt_pad_idx).float() * -1e9

        tgt_len = tgt.shape[1]
        # The look-ahead mask is already a float tensor
        tgt_sub_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(
            tgt.device
        )

        # Return both masks separately
        return tgt_pad_mask, tgt_sub_mask

    def forward(self, src, tgt):
        # generate masks
        src_mask = self.make_src_mask(src)
        tgt_pad_mask, tgt_sub_mask = self.make_tgt_mask(tgt)

        src_emb = self.src_emb(src)
        tgt_emb = self.tgt_emb(tgt)
        src_emb_with_pos = self.positional_encoding(src_emb)
        tgt_emb_with_pos = self.positional_encoding(tgt_emb)

        output = self.transformer(
            src_emb_with_pos,
            tgt_emb_with_pos,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_sub_mask,
        )

        return self.fc_out(output)

    def encode(self, src):
        """Processes the source sequence through the encoder."""
        src_mask = self.make_src_mask(src)
        src_emb = self.positional_encoding(self.src_emb(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, tgt, memory):
        """Processes the target sequence through the decoder, given the encoder output."""
        tgt_pad_mask, tgt_sub_mask = self.make_tgt_mask(tgt)
        tgt_emb = self.positional_encoding(self.tgt_emb(tgt))
        return self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_sub_mask,
        )
