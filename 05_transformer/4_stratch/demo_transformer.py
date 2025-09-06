import torch
from torchmetrics.text.bleu import BLEUScore
from transformer_model import TransformerModel
from train import train
from evaluate import evaluate
from inference import translate, greedy_decode
from preprocess import special_tokens, preprocess, src_vocab, tgt_vocab, train_dataset

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


num_epochs = 20
batch_size = 128
d_model = 256
nhead = 8
dim_feedforward = 512
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
batch_first = True

train_dataloader, valid_dataloader, test_dataset = preprocess(
    batch_size, device, batch_first
)

model = TransformerModel(
    num_encoder_layers,
    num_decoder_layers,
    d_model,
    nhead,
    len(src_vocab),
    len(tgt_vocab),
    dim_feedforward,
    dropout,
    batch_first=batch_first,
).to(device)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=special_tokens["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, optimizer, train_dataloader, loss_fn, device, batch_first)
    valid_loss = evaluate(model, valid_dataloader, loss_fn, device, batch_first)
    print(
        f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Validation loss: {valid_loss:.4f}"
    )


print("Testing Translation on First 32 Samples")
for i in range(32):
    en_sentence = test_dataset[i]["en"]
    de_reference = test_dataset[i]["de"]

    translated = translate(model, en_sentence, device, greedy_decode, batch_first)
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

    translated = translate(model, en_sentence, device, greedy_decode, batch_first)

    all_predictions.append(translated)
    all_references.append([de_reference.lower()])

bleu_metric = BLEUScore()
bleu_score = bleu_metric(all_predictions, all_references)
print(f"Corpus BLEU Score: {bleu_score.item():.4f}")
