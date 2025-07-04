import collections
import numpy
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


assert tokenize("Hello, world!") == ["hello", "world"]


def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenize(text)


counter = collections.Counter()

for tokens in yield_tokens(train_data):
    counter.update(tokens)

vocab = counter.most_common(1000)
print(vocab[:5])

word_to_index = {word: idx + 2 for idx, (word, _) in enumerate(vocab)}
word_to_index["<PAD>"] = 0
word_to_index["<UNK>"] = 1


class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data, word_to_index):
        self.data = data
        self.word_to_index = word_to_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        text_idx = [
            self.word_to_index.get(word, self.word_to_index["<UNK>"])
            for word in tokenize(text)
        ]
        return torch.tensor(text_idx), torch.tensor(label)


train_dataset = IMDBDataset(train_data, word_to_index)
print(train_dataset[0])


def collate_fn(batch):
    texts, labels = zip(*batch)

    max_len = max(len(text) for text in texts)

    padded_texts = [
        torch.cat(
            [text, torch.tensor([word_to_index["<PAD>"]] * (max_len - len(text)))]
        )
        for text in texts
    ]

    padded_texts_tensor = torch.stack(padded_texts).long()

    labels_tensor = torch.tensor(labels).long()

    return padded_texts_tensor, labels_tensor


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)


class TextClassificationModel(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TextClassificationModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Linear(embedding_dim, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = self.embedding(x)
        # [batch_size, embedding_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.pool(x)
        # [batch_size, embedding_dim]
        x = x.squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


embedding_dim = 16
vocab_size = len(word_to_index)
model = TextClassificationModel(vocab_size, embedding_dim)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        outputs = model(texts)
        loss = criterion(outputs.squeeze(), labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")


if not os.path.exists("temp"):
    os.makedirs("temp")

embedding_weights = model.embedding.weight.data.cpu().numpy()
numpy.savetxt("temp/embedding_vectors.tsv", embedding_weights, delimiter="\t")

with open("temp/embedding_labels.tsv", "w") as f:
    for word in word_to_index:
        f.write(f"{word}\n")

print('You can open the two TSV files at: https://projector.tensorflow.org/')
