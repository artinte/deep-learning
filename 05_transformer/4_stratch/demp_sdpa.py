import torch
from torchmetrics.text.bleu import BLEUScore
from preprocess import preprocess

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


num_epochs = 40
batch_size = 128
d_model = 256
num_head = 8
dim_feedforward = 512
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
batch_first = True

train_dataloader, valid_dataloader, test_dataset = preprocess(
    batch_size, device, batch_first
)


