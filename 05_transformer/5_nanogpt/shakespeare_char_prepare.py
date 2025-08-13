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

data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

data_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(data_dir, exist_ok=True)
input_file_path = os.path.join(data_dir, "input.txt")

# download the tiny shakespeare dataset
if not os.path.exists(input_file_path):
    with open(input_file_path, "w") as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, "r") as f:
    data = f.read()

print("Length of dataset in characters:", len(data))

# get all the unique characters taht occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("All the unique characters:", "".join(chars))
print(f"Vocab size: {vocab_size}")

# Create a mapping from characters to integers.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    # encoder: take a string, output a list of integers
    return [stoi[c] for c in s]


def decode(l):
    # decoder: take a list integers, output a string
    return " ".join([itos[i] for i in l])
