import torch
import torchtext
from model import TransformerModel
from data_preprocess import preprocess
from train import train
from evaluate import evaluate
from inference import translate_one_batch

if __name__ == "__main__":
    SRC, TRG, train_dataset, valid_dataset, test_dataset = preprocess()

    src_vocab_size = len(SRC.vocab)
    trg_vocab_size = len(TRG.vocab)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iterator, valid_iterator, test_iterator = (
        torchtext.data.BucketIterator.splits(
            (train_dataset, valid_dataset, test_dataset),
            batch_size=32,
            device=device,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
        )
    )

    for batch in train_iterator:
        src = batch.src
        trg = batch.trg

        print(src.shape)
        print(trg.shape)
        break

    model = TransformerModel(src_vocab_size, trg_vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

    criterion.to(device)

    for epoch in range(10):
        train_loss = train(model, train_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_iterator, criterion)
        print(
            f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}"
        )

    def print_translations(translations):
        for translation in translations:
            print(f"Source: {translation['src']}")
            print(f"Prediction: {translation['hyp']}")
            print(f"Reference: {translation['trg']}")
            print("-" * 50)

    translations = translate_one_batch(model, test_iterator, SRC, TRG)
    print_translations(translations)
