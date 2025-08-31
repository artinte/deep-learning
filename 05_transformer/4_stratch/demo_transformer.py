import math
import torch
import datasets
import torch.nn as nn
import torch.optim as optim
import spacy
from collections import defaultdict
from torch.utils.data import DataLoader
from torchmetrics.text.bleu import BLEUScore
from positional_encoding import PositionalEncoding


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_epochs = 50
batch_size = 128
d_model = 256
num_head = 8
dim_feedforward = 512
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
src_language = "en"
tgt_language = "de"
special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
unk_token_id, pad_token_id, bos_token_id, eos_token_id = 0, 1, 2, 3


try:
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")
except IOError:
    print("Spacy models not found. Downloading...")
    from spacy.cli import download

    download("en_core_web_sm")
    download("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    spacy_de = spacy.load("de_core_news_sm")

token_transform = {
    src_language: lambda text: [
        token.text.lower() for token in spacy_en.tokenizer(text)
    ],
    tgt_language: lambda text: [
        token.text.lower() for token in spacy_de.tokenizer(text)
    ],
}


def build_vocab(data_iter, language, min_freq=2, specials=None):
    if specials is None:
        specials = []

    counts = defaultdict(int)
    for example in data_iter:
        tokens = token_transform[language](example[language])
        for token in tokens:
            counts[token] += 1

    str_to_idx = {token: i for i, token in enumerate(specials)}
    current_idx = len(specials)

    for token, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        if count >= min_freq:
            str_to_idx[token] = current_idx
            current_idx += 1

    idx_to_str = {idx: token for token, idx in str_to_idx.items()}

    def lookup_token(token):
        return str_to_idx.get(token, unk_token_id)

    return str_to_idx, idx_to_str, lookup_token


train_dataset, valid_dataset, test_dataset = datasets.load_dataset(
    "bentrevett/multi30k", split=["train", "validation", "test"]
)

src_vocab, src_rev_vocab, src_lookup = build_vocab(
    train_dataset, src_language, min_freq=2, specials=special_tokens
)
tgt_vocab, tgt_rev_vocab, tgt_lookup = build_vocab(
    train_dataset, tgt_language, min_freq=2, specials=special_tokens
)


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat(
        (
            torch.tensor([bos_token_id]),
            torch.tensor(token_ids),
            torch.tensor([eos_token_id]),
        )
    )


text_transform = {
    src_language: sequential_transforms(
        token_transform[src_language],
        lambda tokens: [src_lookup(token) for token in tokens],
        tensor_transform,
    ),
    tgt_language: sequential_transforms(
        token_transform[tgt_language],
        lambda tokens: [tgt_lookup(token) for token in tokens],
        tensor_transform,
    ),
}


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(text_transform[src_language](item[src_language]))
        tgt_batch.append(text_transform[tgt_language](item[tgt_language]))

    src_batch = nn.utils.rnn.pad_sequence(
        src_batch, padding_value=pad_token_id, batch_first=False
    )
    tgt_batch = nn.utils.rnn.pad_sequence(
        tgt_batch, padding_value=pad_token_id, batch_first=False
    )

    return src_batch, tgt_batch


train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=collate_fn
)
valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, collate_fn=collate_fn
)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.d_model = d_model
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, batch_first=False)

    def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(
            src=src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self, tgt, memory, memory_key_padding_mask, tgt_mask, tgt_key_padding_mask
    ):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(
            tgt_emb,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == pad_token_id).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_token_id).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, dataloader, loss_fn):
    model.train()
    losses = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        )

        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        # Add the gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        losses += loss.item()
    return losses / len(dataloader)


def evaluate(model, dataloader, loss_fn):
    model.eval()
    losses = 0
    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        with torch.no_grad():
            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
            )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(dataloader)


torch.manual_seed(0)
transformer = Seq2SeqTransformer(
    num_encoder_layers,
    num_decoder_layers,
    d_model,
    num_head,
    len(src_vocab),
    len(tgt_vocab),
    dim_feedforward,
    dropout,
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
optimizer = optim.Adam(transformer.parameters(), lr=0.0005)

for epoch in range(1, num_epochs + 1):
    train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
    valid_loss = evaluate(transformer, valid_dataloader, loss_fn)
    print(
        f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}"
    )


def greedy_decode(model, src_sentence, max_len=50):
    model.eval()
    src = text_transform[src_language](src_sentence).to(device).unsqueeze(1)
    src_padding_mask = (src == pad_token_id).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(bos_token_id).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(
                device
            )
            tgt_padding_mask = (ys == pad_token_id).transpose(0, 1)
            out = model.decode(
                ys,
                memory,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

        prob = model.generator(out[-1, :, :])
        next_word_idx = torch.argmax(prob, dim=1).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=0
        )

        if next_word_idx == eos_token_id:
            break

    return ys.flatten()


def topk_decode(model, src_sentence, max_len=50, k=5):
    """
    Decodes a source sentence using top-k sampling.
    """
    model.eval()
    src = text_transform[src_language](src_sentence).to(device).unsqueeze(1)
    src_padding_mask = (src == pad_token_id).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(bos_token_id).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(
                device
            )
            tgt_padding_mask = (ys == pad_token_id).transpose(0, 1)
            out = model.decode(
                ys,
                memory,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

        prob = model.generator(out[-1, :, :])

        # Get the top-k probabilities and indices
        topk_probs, topk_indices = torch.topk(prob, k, dim=1)

        # Sample one from the top-k indices
        next_word_idx = topk_indices.gather(
            1, torch.multinomial(topk_probs, num_samples=1)
        ).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=0
        )

        if next_word_idx == eos_token_id:
            break

    return ys.flatten()


def translate(model, src_sentence, decode_fn=greedy_decode):
    tgt_tokens = decode_fn(model, src_sentence).cpu().numpy()

    def lookup_tokens(indices):
        return [tgt_rev_vocab.get(i, "<unk>") for i in indices]

    return (
        " ".join(lookup_tokens(list(tgt_tokens)))
        .replace("<bos>", "")
        .replace("<eos>", "")
        .strip()
    )


print("Testing Translation on First 32 Samples")
for i in range(32):
    en_sentence = train_dataset[i]["en"]
    de_reference = train_dataset[i]["de"]

    translated = translate(transformer, en_sentence)
    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {translated}")
    print(f"Reference: {de_reference}")


print("-" * 50)
print("Calculating Corpus BLEU Score...")
all_predictions = []
all_references = []
for sample in train_dataset.take(100):
    en_sentence = sample["en"]
    de_reference = sample["de"]

    translated = translate(transformer, en_sentence)

    all_predictions.append(translated)
    all_references.append([de_reference.lower()])

bleu_metric = BLEUScore()
bleu_score = bleu_metric(all_predictions, all_references)
print(f"Corpus BLEU Score: {bleu_score.item():.4f}")
