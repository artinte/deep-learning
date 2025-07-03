import collections
import os
import pathlib
import re
import sys
import string
import shutil
import torch
import torchtext

print(torch.__version__)
print(torchtext.__version__)

project_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from common import imdb_text

dataset_dir = imdb_text.get_dataset_dir()
print(os.listdir(dataset_dir))

train_dir = os.path.join(dataset_dir, "train")
print(os.listdir(train_dir))

remove_dir = os.path.join(train_dir, "unsup")
shutil.rmtree(remove_dir)

# Embed a 1000 word vocabulary into 5 dimensions.
simple_embedding_layer = torch.nn.Embedding(num_embeddings=1000, embedding_dim=5)
result = simple_embedding_layer(torch.tensor([1, 2, 3]))
assert result.shape == (3, 5)
print(result)

embedding_matrix = torch.randn(1000, 5)
result = embedding_matrix[[1, 2, 3]]
assert result.shape == (3, 5)


result = simple_embedding_layer(torch.tensor([[0, 1, 2], [3, 4, 5]]))
assert result.shape == (2, 3, 5)


def load_imdb_data(data_dir):
    data = []
    for label_type in ["pos", "neg"]:
        dir_path = os.path.join(data_dir, label_type)
        label = 1 if label_type == "pos" else 0
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), encoding="utf-8") as f:
                    text = f.read()
                    data.append((text, label))
    return data


train_data = load_imdb_data(train_dir)


def custom_standardization(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)  # replace <br /> or <br> with space
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    return text


def tokenize(text):
    # Tokenizer function (simple whitespace tokenizer)
    return custom_standardization(text).split()


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenize(text)


# In newer torchtext version, we can use maxtokens to limit token's number.
vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(train_data))
print(list(vocab.stoi.keys())[:10])
