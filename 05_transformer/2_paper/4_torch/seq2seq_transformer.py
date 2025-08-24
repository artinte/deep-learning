import torch

class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self,src_vocab_size,
                           tgt_vocab_size,
                           d_model,
                           n_head,
                           num_encoder_layers,
                           num_decoder_layers,
                           dim_feedforward,
                           dropout):
        super(Seq2SeqTransformer, self).__init__()