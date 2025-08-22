import torch
from positional_encoding import PositionalEncoding

class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        embed_size=256,
        nhead=8,
        n_ff=1024,
        nlayers=6,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()

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

    def forward(self, src, trg):
        src_emb = self.src_emb(src)
        trg_emb = self.trg_emb(trg)
        src_emb_with_pos = self.positional_encoding(src_emb)
        trg_emb_with_pos = self.positional_encoding(trg_emb)

        output = self.transformer(src_emb_with_pos, trg_emb_with_pos)

        return self.fc_out(output)
