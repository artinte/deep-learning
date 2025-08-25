import torch
from model import TransformerModel
from data_preprocess import preprocess
from train import train
from evaluate import evaluate
from inference import translate_one_batch


def print_translations(translations):
    for tran in translations:
        print("-" * 50)
        print(f"Source: {tran['src']}")
        print(f"Prediction: {tran['hyp']}")
        print(f"Reference: {tran['tgt']}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_field, tgt_field, train_iter, valid_iter, test_iter = preprocess(device)
    src_vocab_size = len(src_field.vocab)
    tgt_vocab_size = len(tgt_field.vocab)
    print(f"Src vocab size: {src_vocab_size}")
    print(f"Tgt vocab size: {tgt_vocab_size}")

    src_pad_token = src_field.vocab.stoi[src_field.pad_token]
    tgt_pad_token = tgt_field.vocab.stoi[tgt_field.pad_token]
    model = TransformerModel(
        src_vocab_size,
        tgt_vocab_size,
        src_pad_token,
        tgt_pad_token,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=tgt_field.vocab.stoi[tgt_field.pad_token]
    ).to(device)

    for epoch in range(20):
        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss = evaluate(model, valid_iter, criterion)
        print(
            f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Val loss: {valid_loss:.3f}"
        )

    translations = translate_one_batch(model, test_iter, src_field, tgt_field)
    print_translations(translations)
