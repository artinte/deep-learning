from collections import defaultdict
from functools import partial
import datasets
import spacy
import torch

src_language = "en"
tgt_language = "de"

special_tokens = {
    "<unk>": 0,
    "<pad>": 1,
    "<bos>": 2,
    "<eos>": 3,
}

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
        return str_to_idx.get(token, special_tokens["<unk>"])

    return str_to_idx, idx_to_str, lookup_token


train_dataset, valid_dataset, test_dataset = datasets.load_dataset(
    "bentrevett/multi30k", split=["train", "validation", "test"]
)

src_vocab, src_rev_vocab, src_lookup = build_vocab(
    train_dataset, src_language, min_freq=2, specials=special_tokens.keys()
)
tgt_vocab, tgt_rev_vocab, tgt_lookup = build_vocab(
    train_dataset, tgt_language, min_freq=2, specials=special_tokens.keys()
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
            torch.tensor([special_tokens["<bos>"]]),
            torch.tensor(token_ids),
            torch.tensor([special_tokens["<eos>"]]),
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


def collate_fn(batch, device):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(text_transform[src_language](item[src_language]))
        tgt_batch.append(text_transform[tgt_language](item[tgt_language]))

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, padding_value=special_tokens["<pad>"], batch_first=False
    )
    tgt_batch = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, padding_value=special_tokens["<pad>"], batch_first=False
    )

    return src_batch.to(device), tgt_batch.to(device)


def preprocess(batch_size, device):
    collate = partial(collate_fn, device=device)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, collate_fn=collate
    )

    return train_dataloader, valid_dataloader, test_dataset
