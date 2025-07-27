import datasets
import math
import transformers
import torch
import random
from matplotlib import pyplot

# 1. Load the dataset
dataset = datasets.load_dataset("ted_hrlr", "pt_to_en", trust_remote_code=True)
train_examples = dataset["train"]
val_examples = dataset["validation"]
test_examples = dataset["test"]

print(f"Train examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")
print(f"Test examples: {len(test_examples)}")

print(train_examples[0])

# 2. Initialize a tokenizer
# We'll use a pre-trained tokenizer suitable for sequence-to-sequence tasks (like translation).
# 'Helsinki-NLP/opus-mt-pt-en' is a good choice for Portuguese to English.
tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = "pt"
tokenizer.tgt_lang = "en"

print(f"Vocab Size: {len(tokenizer)}")

text = "Hello, how are you?"
encoded = tokenizer(text, return_tensors="pt")
print("Token IDs:", encoded["input_ids"])
print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))


# 3. Create a custom PyTorch Dataset class
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        pair = self.examples[idx]["translation"]
        source_text = pair["pt"]
        target_text = pair["en"]

        # Tokenize source and target texts
        # add_special_tokens=True adds [CLS] and [SEP] tokens
        # truncation=True truncates sequences longer than max_length
        # padding='max_length' pads sequences shorter than max_length
        # return_tensors='pt' returns PyTorch tensors
        tokenized_source = self.tokenizer(
            source_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized_target = self.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Remove the batch dimension (squeeze) as __getitem__ expects single examples
        return {
            "input_ids": tokenized_source["input_ids"].squeeze(0),
            "attention_mask": tokenized_source["attention_mask"].squeeze(0),
            # For translation, target input_ids are often used as labels
            "labels": tokenized_target["input_ids"].squeeze(0),
        }


# 4. Instantiate your PyTorch datasets
max_seq_length = 128  # You can adjust this based on your data and model
train_dataset_pt = TranslationDataset(
    train_examples, tokenizer, max_length=max_seq_length
)
val_dataset_pt = TranslationDataset(val_examples, tokenizer, max_length=max_seq_length)

print(f"PyTorch Training Dataset size: {len(train_dataset_pt)}")
print(f"PyTorch Validation Dataset size: {len(val_dataset_pt)}")

# Example of accessing an item from the PyTorch dataset
sample_idx = random.randint(0, 100)
sample_item = train_dataset_pt[sample_idx]
print("Sample item from PyTorch training dataset:")
print(f"Input IDs shape: {sample_item['input_ids'].shape}")
print(f"Attention Mask shape: {sample_item['attention_mask'].shape}")
print(f"Labels shape: {sample_item['labels'].shape}")

original_src_text = train_examples[sample_idx]["translation"]["pt"]
print(f"Original Source Text (PT): {original_src_text}")

src_token_ids = sample_item["input_ids"].tolist()
print(f"Source Token IDs: {src_token_ids}")

attention_mask_values = sample_item["attention_mask"].tolist()
print(f"Attention  Mask Values: {attention_mask_values}")

tgt_token_ids = sample_item["labels"].tolist()
print(f"Target (Label) Token IDs: {tgt_token_ids}")

# Convert source token IDs to text.
src_text = tokenizer.decode(src_token_ids, skip_special_tokens=True)
print(f"Source Text: {src_text}")
tgt_text = tokenizer.decode(tgt_token_ids, skip_special_tokens=True)
print(f"Target (Label) Text: {tgt_text}")

train_dataloader = torch.utils.data.DataLoader(
    train_dataset_pt, batch_size=16, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset_pt, batch_size=16, shuffle=False
)

print(f"Number of batches in training DataLoader: {len(train_dataloader)}")
print(f"Number of batches in validation DataLoader: {len(val_dataloader)}")

# Example of iterating through a batch
sample_batch = next(iter(train_dataloader))
print("Sample batch from DataLoader:")
print(f"Input IDs batch shape: {sample_batch['input_ids'].shape}")
print(f"Attention Mask batch shape: {sample_batch['attention_mask'].shape}")
print(f"Labels batch shape: {sample_batch['labels'].shape}")


def positional_encoding(length, depth):
    depth = depth // 2

    # (seq, 1)
    positions = torch.arange(length).unsqueeze(1)
    # (1, depth)
    depths = torch.arange(int(depth)).unsqueeze(0) / depth
    # (1, depth)
    angle_rates = 1 / (10000**depths)
    # (pos, depth)
    angle_rads = positions * angle_rates

    pos_encoding = torch.cat([torch.sin(angle_rads), torch.cos(angle_rads)], axis=-1)
    return pos_encoding.float()


pe = positional_encoding(length=100, depth=64)
assert pe.shape == (100, 64)
for i in range(5):
    pyplot.plot(pe[:, i].numpy(), label=f"dim {i}")

pyplot.xlabel("Position")
pyplot.ylabel("Encoding Value")
pyplot.legend()
pyplot.grid(True)
pyplot.show()


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d_model, pad_token_id, max_len=2048):
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token_id
        )
        pe = positional_encoding(length=max_len, depth=d_model)
        self.register_buffer("pos_encoding", pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x_embed = self.embedding(x) * math.sqrt(self.d_model)
        x_embed = x_embed + self.pos_encoding[:seq_len, :].unsqueeze(dim=0)
        return x_embed


pe_layer = PositionalEmbedding(
    vocab_size=len(tokenizer), d_model=512, pad_token_id=tokenizer.pad_token_id
)

sample_output = pe_layer(sample_batch["input_ids"])
# (batch_size, seq_len, d_model)
print(sample_output.shape)


class BaseAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__()
        self.mha = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
            **kwargs,
        )
        self.layernorm = torch.nn.LayerNorm(normalized_shape=d_model)
