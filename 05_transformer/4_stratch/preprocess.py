import datasets
import nltk
import torch
from collections import Counter
from functools import partial

unknown_token = "<unk>"
pad_token = "<pad>"
sos_token = "<sos>"
eos_token = "<eos>"


class Vocabulary:
    def __init__(self, specials, min_freq=1):
        self.specials = specials
        self.min_freq = min_freq
        self.stoi = {token: i for i, token in enumerate(specials)}
        self.itos = {i: token for i, token in enumerate(specials)}
        self.unk_idx = self.stoi[unknown_token]

    def build_vocab(self, tokens_list):
        counter = Counter(tokens_list)
        # Sort by frequency and then alphabetically for consistent ordering
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        for token, freq in sorted_tokens:
            if freq >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.stoi)

        self.itos = {i: token for token, i in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def numericalize(self, tokens):
        return [self.stoi.get(token, self.unk_idx) for token in tokens]


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_len=None):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # Tokenize using nltk
        src_tokens = nltk.tokenize.word_tokenize(src_text) + [eos_token]
        tgt_tokens = [sos_token] + nltk.tokenize.word_tokenize(tgt_text) + [eos_token]

        # Numericalize tokens
        src_numerical = self.src_vocab.numericalize(src_tokens)
        tgt_numerical = self.tgt_vocab.numericalize(tgt_tokens)

        return {"src": src_numerical, "tgt": tgt_numerical}


# Custom collate function for DataLoader to handle padding
def collate_fn_with_vocab(batch, src_vocab, tgt_vocab):
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]

    # Pad sequences to the length of the longest in the batch
    src_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(s) for s in src_list],
        batch_first=True,
        padding_value=src_vocab.stoi[pad_token],
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t) for t in tgt_list],
        batch_first=True,
        padding_value=tgt_vocab.stoi[pad_token],
    )

    # Create key padding masks
    src_mask = src_padded == src_vocab.stoi[pad_token]
    tgt_mask = tgt_padded == tgt_vocab.stoi[pad_token]

    return src_padded, tgt_padded, src_mask, tgt_mask


def preprocess(device, batch_size=32, dataset=None):
    nltk.download("punkt")
    if dataset == None:
        dataset = datasets.load_dataset("bentrevett/multi30k")
    print(dataset)
    # {'en': 'Two young, White males are outside near many bushes.',
    # 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
    print(f"First train sample: {dataset["train"][0]}")

    train_data = [(example["de"], example["en"]) for example in dataset["train"]]
    valid_data = [(example["de"], example["en"]) for example in dataset["validation"]]
    test_data = [(example["de"], example["en"]) for example in dataset["test"]]

    # Create vocabulary instances
    src_vocab = Vocabulary(specials=[unknown_token, pad_token, eos_token], min_freq=2)
    tgt_vocab = Vocabulary(
        specials=[unknown_token, pad_token, sos_token, eos_token], min_freq=2
    )

    collate_fn = partial(
        collate_fn_with_vocab, src_vocab=src_vocab, tgt_vocab=tgt_vocab
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    print(f"Source <eos> index: {src_vocab.stoi[eos_token]}")
    print(f"Source <pad> index: {src_vocab.stoi[pad_token]}")
    print(f"Source <unk> index: {src_vocab.stoi[unknown_token]}")
    print(f"Target <sos> index: {tgt_vocab.stoi[sos_token]}")
    print(f"Target <eos> index: {tgt_vocab.stoi[eos_token]}")
    print(f"Target <pad> index: {tgt_vocab.stoi[pad_token]}")
    print(f"Target <unk> index: {tgt_vocab.stoi[unknown_token]}")

    print("Source vocabulary size: " + str(len(src_vocab)))
    print("Target vocabulary size: " + str(len(tgt_vocab)))

    print(f"Source vocab examples: {list(src_vocab)[:20]}")
    print(f"Target vocab examples: {list(tgt_vocab)[:20]}")

    for src, tgt, src_mask, tgt_mask in test_dataloader:
        print(f"First src train sample shape: {src.shape}")
        print(f"First tgt train sample shape: {tgt.shape}")
        print(f"First src train sample: {src}")
        print(f"First tgt train sample: {tgt}")
        print(f"First src train mask shape: {src_mask.shape}")
        print(f"First tgt train mask shape: {tgt_mask.shape}")
        print(f"First src train mask: {src_mask}")
        print(f"First tgt train mask: {tgt_mask}")
        break

    return (
        src_vocab,
        tgt_vocab,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        test_data,
    )
