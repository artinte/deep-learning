import os
import pathlib
import sys
import shutil
import torch

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
