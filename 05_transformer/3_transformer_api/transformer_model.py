import torch
from positional_encoding import PositionalEncoding


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
        batch_first,
    ):
        super(TransformerModel, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.d_model = d_model
        self.generator = torch.nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, batch_first=batch_first
        )

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(
            src=src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self, tgt, memory, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask
    ):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(
            tgt_emb,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
