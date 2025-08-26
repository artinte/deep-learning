import torch
from preprocess import preprocess
from tokenizer import get_tokenizer
from train import train
from evaluate import evaluate
from inference import greedy_translate, translate_beam_search
from transformer_model import TransformerModel

torch.set_printoptions(profile="full")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = get_tokenizer()

train_dataloader, valid_dataloader, test_dataloader, data_test = preprocess(tokenizer, device)

src_vocab_size = tokenizer.vocab_size
tgt_vocab_size = tokenizer.vocab_size
d_model = 512
n_head = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.2
num_epochs = 20

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
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

print("Starting model training...")
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    valid_loss = evaluate(model, valid_dataloader, criterion)
    print(
        f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}"
    )


print("Testing Translation on First 32 Samples")
for i in range(32):
    en_sentence = data_test[i]["en"]
    de_reference = data_test[i]["de"]

    # translated = greedy_translate(model, en_sentence, tokenizer)
    translated = translate_beam_search(model, en_sentence, tokenizer)
    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {translated}")
    print(f"Reference: {de_reference}")
