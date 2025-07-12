import torch
import nltk
import torch
import math

import datasets
from torchtext.data import Field, Example, Dataset, BucketIterator
from nltk.tokenize import word_tokenize


nltk.download("punkt")
nltk.download("punkt_tab")

dataset = datasets.load_dataset("bentrevett/multi30k")
print(dataset)
# {'en': 'Two young, White males are outside near many bushes.',
# 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
print(dataset["train"][0])

train_data = [(example["de"], example["en"]) for example in dataset["train"]]
valid_data = [(example["de"], example["en"]) for example in dataset["validation"]]
test_data = [(example["de"], example["en"]) for example in dataset["test"]]

SRC = Field(
    tokenize=word_tokenize,
    init_token="<sos>",
    eos_token="<eos>",
    pad_token="<pad>",
    lower=True,
    batch_first=True,
)
TRG = Field(
    tokenize=word_tokenize,
    init_token="<sos>",
    eos_token="<eos>",
    pad_token="<pad>",
    lower=True,
    batch_first=True,
)

train_examples = [
    Example.fromlist([src, trg], fields=[("src", SRC), ("trg", TRG)])
    for src, trg in train_data
]
valid_examples = [
    Example.fromlist([src, trg], fields=[("src", SRC), ("trg", TRG)])
    for src, trg in valid_data
]
test_examples = [
    Example.fromlist([src, trg], fields=[("src", SRC), ("trg", TRG)])
    for src, trg in test_data
]

train_dataset = Dataset(examples=train_examples, fields=[("src", SRC), ("trg", TRG)])
valid_dataset = Dataset(examples=valid_examples, fields=[("src", SRC), ("trg", TRG)])
test_dataset = Dataset(examples=test_examples, fields=[("src", SRC), ("trg", TRG)])

SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

print("Source vocabulary size: " + str(len(SRC.vocab)))
print("Target vocabulary size: " + str(len(TRG.vocab)))

print([word for word, _ in list(SRC.vocab.stoi.items())[:10]])
print([word for word, _ in list(TRG.vocab.stoi.items())[:10]])

src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
)


BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
)

for batch in train_iterator:
    src = batch.src
    trg = batch.trg

    print(src.shape)
    print(trg.shape)
    break


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, emb_size, nhead, dropout=0.1):
        super().__init__()
        assert emb_size % nhead == 0
        self.d_k = emb_size // nhead
        self.nhead = nhead

        self.q_linear = torch.nn.Linear(emb_size, emb_size)
        self.k_linear = torch.nn.Linear(emb_size, emb_size)
        self.v_linear = torch.nn.Linear(emb_size, emb_size)
        self.out_linear = torch.nn.Linear(emb_size, emb_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, T_q, C = query.shape
        T_k = key.size(1)
        T_v = value.size(1)
        H = self.nhead
        d_k = self.d_k

        # (B, H, T_q, d_k)
        q = (self.q_linear(query).view(B, T_q, H, d_k).transpose(1, 2))
        # (B, H, T_k, d_k)
        k = self.k_linear(key).view(B, T_k, H, d_k).transpose(1, 2)
        # (B, H, T_v, d_k)
        v = (self.v_linear(value).view(B, T_v, H, d_k).transpose(1, 2))
        # (B, H, T_q, T_k)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            assert (
                attn_scores.shape[-2:] == mask.shape[-2:]
            ), f"attn_scores {attn_scores.shape} vs mask {mask.shape}"
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = attn_weights @ v  # (B, H, T_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T_q, H * d_k)  # (B, T_q, C)
        return self.out_linear(out)


class FeedForward(torch.nn.Module):
    def __init__(self, emb_size, hidden_dim, dropout=0.1):
        super().__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(emb_size, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, emb_size),
        )

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(torch.nn.Module):
    def __init__(self, emb_size, nhead, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_size, nhead, dropout)
        self.ff = FeedForward(emb_size, hidden_dim, dropout)
        self.norm1 = torch.nn.LayerNorm(emb_size)
        self.norm2 = torch.nn.LayerNorm(emb_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = x + self.dropout(
            self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=src_mask)
        )
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, emb_size, nhead, hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_size, nhead, dropout)
        self.cross_attn = MultiHeadAttention(emb_size, nhead, dropout)
        self.ff = FeedForward(emb_size, hidden_dim, dropout)

        self.norm1 = torch.nn.LayerNorm(emb_size)
        self.norm2 = torch.nn.LayerNorm(emb_size)
        self.norm3 = torch.nn.LayerNorm(emb_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        x = x + self.dropout(
            self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=tgt_mask)
        )
        x = x + self.dropout(
            self.cross_attn(self.norm2(x), memory, memory, mask=src_mask)
        )
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x


class Encoder(torch.nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer for _ in range(N)])
        self.norm = torch.nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer for _ in range(N)])
        self.norm = torch.nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


class Encoder(torch.nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer for _ in range(N)])
        self.norm = torch.nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(torch.nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = torch.nn.ModuleList([layer for _ in range(N)])
        self.norm = torch.nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        emb_size=256,
        nhead=8,
        nhid=1024,
        nlayers=6,
        dropout=0.1,
    ):
        super(TransformerModel, self).__init__()

        self.src_emb = torch.nn.Embedding(src_vocab_size, emb_size)
        self.trg_emb = torch.nn.Embedding(trg_vocab_size, emb_size)

        self.encoder = Encoder(EncoderLayer(emb_size, nhead, nhid, dropout), nlayers)
        self.decoder = Decoder(DecoderLayer(emb_size, nhead, nhid, dropout), nlayers)

        self.fc_out = torch.nn.Linear(emb_size, trg_vocab_size)

    def forward(self, src, tgt, src_mask=None):
        src_emb = self.src_emb(src)
        tgt_emb = self.trg_emb(tgt)

        memory = self.encoder(src_emb, src_mask)
        tgt_mask = self.generate_square_subsequent_mask(tgt_emb.size(1), tgt.device)

        output = self.decoder(tgt_emb, memory, tgt_mask, src_mask)
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz, device):
        """
        sz: target sequence length (T)
        Returns a mask of shape (1, 1, T, T) for broadcasting to (B, H, T_q, T_k)
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        mask = ~mask  # flip to True: allowed, False: masked
        return mask.unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)


def make_model(src_vocab_len, trg_vocab_len):
    model = TransformerModel(src_vocab_len, trg_vocab_len)
    return model


model = make_model(src_vocab_size, trg_vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

criterion.to(device)

EPOCHS = 40
CLIP = 1.0  # Gradient clipping


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        assert src.size(0) == trg.size(
            0
        ), f"src batch size {src.size(0)} does not match trg batch size {trg.size(0)}"

        optimizer.zero_grad()

        # Exclude the last token from target sequence (shifted target sequence)
        output = model(src, trg[:, :-1])

        output_dim = output.shape[-1]

        # Flatten output and trg to calculate loss
        output = output.view(-1, output_dim)  # (batch_size * seq_len, trg_vocab_size)
        trg = (
            trg[:, 1:].contiguous().view(-1)
        )  # Exclude first token from target sequence

        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])

            output_dim = output.shape[-1]

            output = output.view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


for epoch in range(EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(
        f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}"
    )


def translate_one_batch(model, iterator, SRC, TRG):
    model.eval()
    translations = []

    with torch.no_grad():
        batch = next(iter(iterator))
        src = batch.src
        trg = batch.trg

        output = model(src, trg[:, :-1])
        output = output.argmax(dim=-1)

        for i in range(src.size(0)):
            src_tokens = [SRC.vocab.itos[idx] for idx in src[i]]
            hyp_tokens = []
            for idx in output[i]:
                if idx == TRG.vocab.stoi[TRG.eos_token]:
                    break
                if idx != TRG.vocab.stoi[TRG.pad_token]:
                    hyp_tokens.append(TRG.vocab.itos[idx])

            translations.append(
                {
                    "src": " ".join(src_tokens),
                    "hyp": " ".join(hyp_tokens),
                    "trg": " ".join(
                        [
                            TRG.vocab.itos[idx]
                            for idx in trg[i][1:].cpu().numpy()
                            if idx != TRG.vocab.stoi[TRG.pad_token]
                        ]
                    ),
                }
            )

    return translations


def print_translations(translations):
    for translation in translations:
        print(f"Source: {translation['src']}")
        print(f"Prediction: {translation['hyp']}")
        print(f"Reference: {translation['trg']}")
        print("-" * 50)


translations = translate_one_batch(model, test_iterator, SRC, TRG)
print_translations(translations)
