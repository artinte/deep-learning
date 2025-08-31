import math
import torch


class PositionalEncoding(torch.nn.Module):
    """
    Injects positional information into the embeddings.
    """

    def __init__(self, d_model, dropout=0.1, max_len=8192, batch_first=True):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # equation term: 10000**(-2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # PE(pos, 2i) = sin(pos / (10000^(2i/d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model)))
        pe[:, 1::2] = torch.cos(position * div_term)
        if self.batch_first:
            # Shape: [1, max_len, d_model]
            pe = pe.unsqueeze(0)
        else:
            # Shape: [max_len, 1, d_model]
            pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if self.batch_first:
            # x has shape [batch_size, seq_len, d_model]
            x = x + self.pe[:, : x.size(1), :].requires_grad_(False)
        else:
            # x has shape [seq_len, batch_size, d_model]
            x = x + self.pe[: x.size(0), :].requires_grad_(False)
        return self.dropout(x)
