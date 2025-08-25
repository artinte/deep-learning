import torch
from transformer_model import TransformerModel
from preprocess import preprocess
from train import train
from evaluate import evaluate
from inference import translate_one_batch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

src_field, tgt_field, train_iter, valid_iter, test_iter = preprocess(device)

src_vocab_size = len(src_field.vocab)
tgt_vocab_size = len(tgt_field.vocab)
d_model = 512
n_head = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.2
num_epochs = 30

src_pad_token = src_field.vocab.stoi[src_field.pad_token]
tgt_pad_token = tgt_field.vocab.stoi[tgt_field.pad_token]
model = TransformerModel(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=d_model,
    n_head=n_head,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    src_pad_idx=src_pad_token,
    tgt_pad_idx=tgt_pad_token,
).to(device)

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=tgt_field.vocab.stoi[tgt_field.pad_token], label_smoothing=0.1
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print("Starting model training...")
for epoch in range(num_epochs):
    train_loss = train(model, train_iter, optimizer, criterion)
    valid_loss = evaluate(model, valid_iter, criterion)
    print(
        f"Epoch {epoch+1} | Train loss: {train_loss:.3f} | Val loss: {valid_loss:.3f}"
    )

translations = translate_one_batch(model, test_iter, src_field, tgt_field)
for tran in translations:
    print("-" * 50)
    print(f"Source: {tran['src']}")
    print(f"Prediction: {tran['hyp']}")
    print(f"Reference: {tran['tgt']}")
