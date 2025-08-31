import math
import torch
import datasets
import torch.nn as nn
import torch.optim as optim
import spacy
from collections import defaultdict
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
d_model = 512
num_head = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
SRC_LANGUAGE = "en"
TGT_LANGUAGE = "de"
SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3


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
    SRC_LANGUAGE: lambda text: [
        token.text.lower() for token in spacy_en.tokenizer(text)
    ],
    TGT_LANGUAGE: lambda text: [
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
        return str_to_idx.get(token, UNK_IDX)

    return str_to_idx, idx_to_str, lookup_token


dataset = datasets.load_dataset("bentrevett/multi30k", "default")
train_dataset = dataset["train"]

src_vocab, src_rev_vocab, src_lookup = build_vocab(
    train_dataset, SRC_LANGUAGE, min_freq=2, specials=SPECIAL_TOKENS
)
tgt_vocab, tgt_rev_vocab, tgt_lookup = build_vocab(
    train_dataset, TGT_LANGUAGE, min_freq=2, specials=SPECIAL_TOKENS
)


def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


def tensor_transform(token_ids):
    return torch.cat(
        (torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX]))
    )


text_transform = {
    SRC_LANGUAGE: sequential_transforms(
        token_transform[SRC_LANGUAGE],
        lambda tokens: [src_lookup(token) for token in tokens],
        tensor_transform,
    ),
    TGT_LANGUAGE: sequential_transforms(
        token_transform[TGT_LANGUAGE],
        lambda tokens: [tgt_lookup(token) for token in tokens],
        tensor_transform,
    ),
}


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for item in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](item[SRC_LANGUAGE]))
        tgt_batch.append(text_transform[TGT_LANGUAGE](item[TGT_LANGUAGE]))

    src_batch = nn.utils.rnn.pad_sequence(
        src_batch, padding_value=PAD_IDX, batch_first=False
    )
    tgt_batch = nn.utils.rnn.pad_sequence(
        tgt_batch, padding_value=PAD_IDX, batch_first=False
    )

    return src_batch, tgt_batch


train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, collate_fn=collate_fn
)
valid_dataloader = DataLoader(
    dataset["validation"], batch_size=batch_size, collate_fn=collate_fn
)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward=512,
        dropout=0.3,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.d_model = d_model
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=0.1)

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
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)),
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

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
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    d_model,
    num_head,
    len(src_vocab),
    len(tgt_vocab),
    FFN_HID_DIM,
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
optimizer = optim.Adam(transformer.parameters(), lr=0.0005)

NUM_EPOCHS = 30
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
    valid_loss = evaluate(transformer, valid_dataloader, loss_fn)
    print(
        f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}"
    )


def greedy_decode(model, src_sentence, max_len=50):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).to(device).unsqueeze(1)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(
                device
            )
            tgt_padding_mask = (ys == PAD_IDX).transpose(0, 1)
            out = model.decode(
                ys,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

        prob = model.generator(out[-1, :, :])
        next_word_idx = torch.argmax(prob, dim=1).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=0
        )

        if next_word_idx == EOS_IDX:
            break

    return ys.flatten()


def translate(model, src_sentence):
    tgt_tokens = greedy_decode(model, src_sentence).cpu().numpy()

    def lookup_tokens(indices):
        return [tgt_rev_vocab.get(i, "<unk>") for i in indices]

    return (
        " ".join(lookup_tokens(list(tgt_tokens)))
        .replace("<bos>", "")
        .replace("<eos>", "")
        .strip()
    )


test_sentences = [
    "A man is playing guitar.",
    "Two dogs are running in the park.",
    "The child is eating an apple.",
    "A man in an orange hat starring at something.",
    "A Boston Terrier is running on lush green grass in front of a white fence.",
    "A girl in karate uniform breaking a stick with a front kick.",
    "Five people wearing winter jackets and helmets stand in the snow, with snowmobiles in the background.",
    "People are fixing the roof of a house.",
]

for s in test_sentences:
    print(f"{SRC_LANGUAGE} : {s}")
    print(f"{TGT_LANGUAGE} : {translate(transformer, s)}")
    print("-" * 40)
