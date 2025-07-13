import argparse
import torch
import os
from torch.distributed.fsdp import fully_shard, FSDPModule
from model import ModelArgs, Transformer

torch.manual_seed(0)
vocab_size = 1024
batch_size = 32
seq_len = 64
model_args = ModelArgs(
    n_layers=10,
    n_heads=4,
    vocab_size=vocab_size,
    max_seq_len=seq_len,
    dropout_p=0,
)

model = Transformer(model_args)
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

assert isinstance(model, Transformer)
assert isinstance(model, FSDPModule)

print(model)

