import math
import torch
import datasets
import torch.nn as nn
import torch.optim as optim
import spacy
from collections import defaultdict
from torch.utils.data import DataLoader
from torchmetrics.text.bleu import BLEUScore
from transformer_model import TransformerModel
from train import train
from evaluate import evaluate
from preprocess import (
    special_tokens,
    src_language,
    tgt_language,
    tgt_rev_vocab,
    train_dataset,
    valid_dataset,
    preprocess,
    text_transform,
)

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

src_vocab, tgt_vocab, train_dataloader, valid_dataloader = preprocess(
    batch_size, device
)

torch.manual_seed(0)
model = TransformerModel(
    num_encoder_layers,
    num_decoder_layers,
    d_model,
    num_head,
    len(src_vocab),
    len(tgt_vocab),
    dim_feedforward,
    dropout,
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_tokens["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(1, num_epochs + 1):
    train_loss = train(
        model, optimizer, train_dataloader, loss_fn, special_tokens["<pad>"], device
    )
    valid_loss = evaluate(
        model, valid_dataloader, loss_fn, special_tokens["<pad>"], device
    )
    print(
        f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}"
    )


def greedy_decode(model, src_sentence, max_len=50):
    model.eval()
    src = text_transform[src_language](src_sentence).to(device).unsqueeze(1)
    src_padding_mask = (src == special_tokens["<pad>"]).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(special_tokens["<bos>"]).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(
                device
            )
            tgt_padding_mask = (ys == special_tokens["<pad>"]).transpose(0, 1)
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

        if next_word_idx == special_tokens["<eos>"]:
            break

    return ys.flatten()


def topk_decode(model, src_sentence, max_len=50, k=5):
    """
    Decodes a source sentence using top-k sampling.
    """
    model.eval()
    src = text_transform[src_language](src_sentence).to(device).unsqueeze(1)
    src_padding_mask = (src == special_tokens["<pad>"]).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(special_tokens["<bos>"]).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(ys.size(0)).to(
                device
            )
            tgt_padding_mask = (ys == special_tokens["<pad>"]).transpose(0, 1)
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

        if next_word_idx == special_tokens["<eos>"]:
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

    translated = translate(model, en_sentence)
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

    translated = translate(model, en_sentence)

    all_predictions.append(translated)
    all_references.append([de_reference.lower()])

bleu_metric = BLEUScore()
bleu_score = bleu_metric(all_predictions, all_references)
print(f"Corpus BLEU Score: {bleu_score.item():.4f}")
