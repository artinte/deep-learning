import torch

class TransformerScratchModel(torch.nn.Module):
  def __init__(self):
    super(TransformerScratchModel, self)
    pass

  def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask,
                src_is_causal=False, tgt_is_causal=True):
    pass

  def encode(self, src, src_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
    pass

  def decode(
        self, tgt, memory, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask
    ):
    pass
