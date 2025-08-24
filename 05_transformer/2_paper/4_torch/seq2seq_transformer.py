import torch
from positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        n_head,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.src_embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Set to True for batch-first tensors
        )

        self.generator = torch.nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask):
        """
        The forward pass for training the model.
        Args:
            src (torch.Tensor): The source sequence tensor. Shape: [batch_size, src_seq_len]
            tgt (torch.Tensor): The target seqeunce tensor. Shape: [batch_size, tgt_seq_len]
        """
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )

        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
        )
        # [batch_size, tgt_seq_len, vocab_size]
        return self.generator(out)

    def encode(self, src, src_key_padding_mask):
        src_emb = self.positional_encoding(self.src_embedding(src))
        return self.transformer.encoder(
            src=src_emb, src_key_padding_mask=src_key_padding_mask
        )

    def decode(self, tgt, memory, tgt_key_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(
            tgt.device
        )

        out = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.generator(out)
