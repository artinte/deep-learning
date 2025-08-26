import transformers
import torch
from preprocess import preprocess

mock_data = {
    "train": [
        {"en": "A small dog is playing.", "de": "Ein kleiner Hund spielt."},
        {"en": "The sun is shining.", "de": "Die Sonne scheint."},
    ],
    "validation": [
        {"en": "A cat is sleeping.", "de": "Eine Katze schläft."},
    ],
    "test": [
        {"en": "A bird is flying.", "de": "Ein Vogel fliegt."},
    ],
}

tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader, valid_dataloader, test_dataloader, data_test = preprocess(
    tokenizer, device=device, batch_size=1, dataset=mock_data
)
assert len(train_dataloader) == 2
assert len(valid_dataloader) == 1
assert len(test_dataloader) == 1
