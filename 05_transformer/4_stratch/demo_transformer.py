import time
import torch
from transformer_model import TransformerModel
from preprocess import preprocess
from train import train
from evaluate import evaluate
from inference import greedy_translate
from torchmetrics.text import BLEUScore


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

src_field, tgt_field, train_iter, valid_iter, test_iter, test_data = preprocess(device)

src_vocab_size = len(src_field.vocab)
tgt_vocab_size = len(tgt_field.vocab)
d_model = 256
n_head = 4
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 1024
dropout = 0.2
num_epochs = 30

model = TransformerModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    n_head=n_head,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    device=device,
).to(device)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=tgt_field.vocab.stoi[tgt_field.pad_token], label_smoothing=0.1
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print("Starting model training...")
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = train(model, train_iter, src_field, tgt_field, optimizer, criterion)
    valid_loss = evaluate(model, valid_iter, src_field, tgt_field, criterion)
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    print(
        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}"
    )

print("Testing Translation on First 32 Samples")
for i in range(32):
    en_sentence = test_data[i]["en"]
    de_reference = test_data[i]["de"]

    translated = greedy_translate(model, en_sentence, src_field, tgt_field)
    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {translated}")
    print(f"Reference: {de_reference}")

print("-" * 50)
print("Calculating Corpus BLEU Score...")
all_predictions = []
all_references = []
for sample in test_data:
    en_sentence = sample["en"]
    de_reference = sample["de"]

    translated = greedy_translate(model, en_sentence, src_field, tgt_field)
    all_predictions.append(translated)
    all_references.append([de_reference])

bleu_metric = BLEUScore()
bleu_score = bleu_metric(all_predictions, all_references)
print(f"Corpus BLEU Score: {bleu_score.item():.4f}")
