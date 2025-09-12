import math
import torch
from positional_encoding import PositionalEncoding


class GPTModel(torch.nn.Transformer):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.1):
        super(GPTModel, self).__init__(
            d_model=ninp, nhead=nhead, dim_feedforward=nhid, num_encoder_layers=nlayers
        )
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)

        self.input_emb = torch.nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = torch.nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        torch.nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        torch.nn.init.zeros_(self.decoder.bias)
        torch.nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = torch.log(torch.tril(torch.ones(len(src), len(src)))).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.input_emb(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        output = self.decoder(output)
        return torch.nn.functional.log_softmax(output, dim=-1)
