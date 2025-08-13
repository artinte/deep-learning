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

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
print(stoi)


def encode(s):
    # encoder: take a string, output a list of integers
    return [stoi[c] for c in s]


def decode(l):
    # decoder: take a list integers, output a string
    return "".join([itos[i] for i in l])


sample_text = "Hello, World!"
print("Tokenize", encode(sample_text))
print("Decode:", decode(encode(sample_text)))


# create the train and val splits
n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"Train has {len(train_ids)} tokens")
print(f"Val has {len(val_ids)} tokens")


# export to bin files
train_ids = numpy.array(train_ids, dtype=numpy.uint16)
val_ids = numpy.array(val_ids, dtype=numpy.uint16)
train_ids.tofile(os.path.join(data_dir, "train.bin"))
val_ids.tofile(os.path.join(data_dir, "val.bin"))


# save the meta information as well, to help us encode/decode later
meta = {
    "vocab_size": vocab_size,
    "itos": itos,
    "stoi": stoi,
}

with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
