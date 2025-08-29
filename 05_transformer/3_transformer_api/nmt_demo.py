import time
import torch
from torchmetrics.text import BLEUScore
from preprocess import preprocess
from tokenizer import get_tokenizer
from train import train
from evaluate import evaluate
from inference import greedy_translate
from transformer_model import TransformerModel
from custom_optimizer import CustomOptimizer

torch.set_printoptions(profile="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = get_tokenizer()

train_dataloader, valid_dataloader, _, data_test = preprocess(tokenizer, device)

# Smaller model parameters are less likely to cause overfitting.
src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = tokenizer.vocab_size
d_model = 512
n_head = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.2
num_epochs = 15

model = TransformerModel(
    src_vocab_size,
    tgt_vocab_size,
    d_model,
    n_head,
    num_encoder_layers,
    num_decoder_layers,
    dim_feedforward,
    dropout,
    device,
).to(device)


criterion = torch.nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
)

base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
warnup_steps = 4000
optimizer = CustomOptimizer(base_optimizer, d_model, warnup_steps)

print("Starting model training...")
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss = train(model, train_dataloader, optimizer, criterion)
    valid_loss = evaluate(model, valid_dataloader, criterion)
    end_time = time.time()
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) % 60)
    print(
        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}"
    )

print("Testing Translation on First 32 Samples")
for i in range(32):
    en_sentence = data_test[i]["en"]
    de_reference = data_test[i]["de"]

    # translated = greedy_translate(model, en_sentence, tokenizer)
    translated = greedy_translate(model, en_sentence, tokenizer)
    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {translated}")
    print(f"Reference: {de_reference}")

print("-" * 50)
print("Calculating Corpus BLEU Score...")
all_predictions = []
all_references = []
for sample in data_test:
    en_sentence = sample["en"]
    de_reference = sample["de"]

    translated = greedy_translate(model, en_sentence, tokenizer)
    all_predictions.append(translated)
    all_references.append([de_reference])

bleu_metric = BLEUScore()
bleu_score = bleu_metric(all_predictions, all_references)
print(f"Corpus BLEU Score: {bleu_score.item():.4f}")
