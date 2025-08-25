import torch
from positional_encoding import PositionalEncoding


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        nhead=8,
        n_ff=1024,
        nlayers=6,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.src_emb = torch.nn.Embedding(src_vocab_size, embed_size)
        self.trg_emb = torch.nn.Embedding(trg_vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size)

        self.transformer = torch.nn.Transformer(
            d_model=embed_size,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=n_ff,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = torch.nn.Linear(embed_size, trg_vocab_size)

    def make_src_mask(self, src):
        # This creates a boolean mask that is passed to src_key_padding_mask
        return src == self.src_pad_idx

    def make_trg_mask(self, trg):
        # Create the padding mask as a float tensor with a large negative value
        trg_pad_mask = (trg == self.trg_pad_idx).float() * -1e9
        
        trg_len = trg.shape[1]
        # The look-ahead mask is already a float tensor
        trg_sub_mask = self.transformer.generate_square_subsequent_mask(trg_len).to(trg.device)
        
        # Return both masks separately
        return trg_pad_mask, trg_sub_mask

    def forward(self, src, trg):
        # generate masks
        src_mask = self.make_src_mask(src)
        trg_pad_mask, trg_sub_mask = self.make_trg_mask(trg)

        src_emb = self.src_emb(src)
        trg_emb = self.trg_emb(trg)
        src_emb_with_pos = self.positional_encoding(src_emb)
        trg_emb_with_pos = self.positional_encoding(trg_emb)

        output = self.transformer(
            src_emb_with_pos,
            trg_emb_with_pos,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=trg_pad_mask,
            tgt_mask=trg_sub_mask,
        )

        return self.fc_out(output)
    
    def encode(self, src):
        """Processes the source sequence through the encoder."""
        src_mask = self.make_src_mask(src)
        src_emb = self.positional_encoding(self.src_emb(src))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_mask)

    def decode(self, trg, memory):
        """Processes the target sequence through the decoder, given the encoder output."""
        trg_pad_mask, trg_sub_mask = self.make_trg_mask(trg)
        trg_emb = self.positional_encoding(self.trg_emb(trg))
        return self.transformer.decoder(
            trg_emb,
            memory,
            tgt_key_padding_mask=trg_pad_mask,
            tgt_mask=trg_sub_mask,
        )
