"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""

import os
import pickle
import requests
import numpy

data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"Vocab size: {vocab_size}")

# Create a mapping from characters to integers.
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

def encode(s):
  # encoder: take a string, output a list of integers
  return [stoi[c] for c in s]

def decode(l):
  # decoder: take a list integers, output a string
  return ' '.join(itos[i] for i in l])
