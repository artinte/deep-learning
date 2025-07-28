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

sample_positional_output = pe_layer(sample_batch["input_ids"])
# (batch_size, seq_len, d_model)
print(sample_positional_output.shape)


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
        # To store the last attention scores for visualization
        self.last_attn_scores = None


class CrossAttention(BaseAttention):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(d_model, num_heads, dropout_rate, **kwargs)

    def forward(self, x, context, key_padding_mask=None):
        # x: (batch, traget_seq_len, d_model)
        # context: (batch, source_seq_len, d_model)
        if key_padding_mask is not None and key_padding_mask.device != x.device:
            key_padding_mask = key_padding_mask.to(x.device)

        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            need_weights=True,
            average_attn_weights=False,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        # Residual connection and layer norm.
        x = x + attn_output
        x = self.layernorm(x)

        return x


cross_attn = CrossAttention(d_model=512, num_heads=4, dropout_rate=0.1)
# target sequence
x = torch.randn(16, 128, 512)
# source sequence (e.g., encoder output)
context = torch.randn(16, 64, 512)

output = cross_attn(x, context)
assert output.shape == (16, 128, 512)


class GlobalSelfAttention(BaseAttention):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(d_model, num_heads, dropout_rate, **kwargs)

    def forward(self, x, key_padding_mask=None):
        # query = key = value = x
        # # x: (batch, seq_len, d_model)
        if key_padding_mask is not None and key_padding_mask.device != x.device:
            key_padding_mask = key_padding_mask.to(x.device)

        attn_output, attn_scores = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
            key_padding_mask=key_padding_mask,
            attn_mask=None,
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        # Residual connection and layer norm.
        x = x + attn_output
        x = self.layernorm(x)

        return x


causal_attn = GlobalSelfAttention(d_model=512, num_heads=4, dropout_rate=0.1)
x = torch.randn(16, 128, 512)
output = causal_attn(x)
assert output.shape == (16, 128, 512)


class CausalSelfAttention(BaseAttention):
    def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
        super().__init__(d_model, num_heads, dropout_rate, **kwargs)

    def forward(self, x, key_padding_mask=None):
        # query = key = value = x
        # x: (batch, seq_len, d_model)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(
            x.size(1)
        ).to(x.device)
        causal_mask = causal_mask == float("-inf")
        attn_output, attn_scores = self.mha(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
        )

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        # Residual connection and layer norm.
        x = x + attn_output
        x = self.layernorm(x)

        return x


causal_attn = CausalSelfAttention(d_model=512, num_heads=4, dropout_rate=0.1)
x = torch.randn(16, 128, 512)
output = causal_attn(x)
assert output.shape == (16, 128, 512)

casual_attn_without_dropout = CausalSelfAttention(
    d_model=512, num_heads=4, dropout_rate=0.0
)
x = torch.randn(16, 128, 512)
dummy_mask = torch.zeros(16, 128, dtype=torch.bool, device=x.device)
dummy_mask[:, -10:] = True
out1 = casual_attn_without_dropout(x[:, :3], key_padding_mask=dummy_mask[:, :3])
out2 = casual_attn_without_dropout(x, key_padding_mask=dummy_mask)[:, :3]
torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-5)
print("Causal self-attention without dropout works as expected.")


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_ff)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear2 = torch.nn.Linear(d_ff, d_model)
        self.layernorm = torch.nn.LayerNorm(normalized_shape=d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x_ff = self.linear1(x)
        x_ff = self.relu(x_ff)
        x_ff = self.dropout(x_ff)
        x_ff = self.linear2(x_ff)

        # Residual connection and layer norm.
        x = x + x_ff
        x = self.layernorm(x)

        return x


ffn = FeedForward(d_model=512, d_ff=2048, dropout_rate=0.1)
x = torch.randn(16, 128, 512)
output = ffn(x)
assert output.shape == (16, 128, 512)


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.self_attn = GlobalSelfAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)

    def forward(self, x, key_padding_mask=None):
        # x: (batch, seq_len, d_model)
        x = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = self.ffn(x)
        return x


sample_encoder_layer = EncoderLayer(
    d_model=512, num_heads=4, d_ff=2048, dropout_rate=0.1
)
x = torch.randn(16, 128, 512)
output = sample_encoder_layer(x)
assert output.shape == (16, 128, 512)


class Encoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout_rate=0.1):
        super().__init__()
        self.pos_embedding = PositionalEmbedding(
            vocab_size=len(tokenizer),
            d_model=d_model,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.enc_layers = torch.nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.layernorm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, key_padding_mask=None):
        # x is token-IDs shape: (batch, seq_len)
        # (batch_size, seq_len, d_model)
        x = self.pos_embedding(x)
        # Add dropout.
        x = self.dropout(x)
        # Apply each encoder layer sequentially
        key_padding_mask = key_padding_mask == 0
        for layer in self.enc_layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        # (batch_size, seq_len, d_model)
        return x


# Instaniate the encoder.
sample_encoder = Encoder(
    d_model=512, num_heads=4, d_ff=2048, num_layers=6, dropout_rate=0.1
)
sample_batch_encoder_input = next(iter(train_dataloader))
encoder_input_ids = sample_batch_encoder_input["input_ids"]
print(encoder_input_ids.shape)  # Should be (batch_size, seq_len)
encoder_attention_mask = sample_batch_encoder_input["attention_mask"]

# Forward pass through the encoder.
output = sample_encoder(encoder_input_ids, encoder_attention_mask)
assert output.shape == (encoder_input_ids.shape[0], encoder_input_ids.shape[1], 512)
print("Encoder forward pass successful with padding mask.")


class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attn = CausalSelfAttention(d_model, num_heads, dropout_rate)
        self.cross_attn = CrossAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForward(d_model, d_ff, dropout_rate)

    def forward(
        self, x, context, x_key_padding_mask=None, context_key_padding_mask=None
    ):
        # x: (batch, seq_len, d_model)
        # context: (batch, source_seq_len, d_model)
        x = self.causal_self_attn(x, key_padding_mask=x_key_padding_mask)
        x = self.cross_attn(x, context, key_padding_mask=context_key_padding_mask)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attn.last_attn_scores

        # Apply the feed-forward network
        x = self.ffn(x)
        return x


sample_decoder_layer = DecoderLayer(
    d_model=512, num_heads=4, d_ff=2048, dropout_rate=0.1
)
# Create a sample input (batch_size=16, seq_len=128)
x = torch.randn(16, 128, 512)
# Create a sample context (batch_size=16, source_seq_len=64)
context = torch.randn(16, 64, 512)
# Forward pass through the decoder layer.
output = sample_decoder_layer(x, context)
assert output.shape == (16, 128, 512)


class Decoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        # To store the last attention scores for visualization
        self.last_attn_scores = None

        self.pos_embedding = PositionalEmbedding(
            vocab_size=len(tokenizer),
            d_model=d_model,
            pad_token_id=tokenizer.pad_token_id,
        )
        self.dec_layers = torch.nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.layernorm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(
        self,
        decoder_input_ids,
        context,
        x_key_padding_mask=None,
        context_key_padding_mask=None,
    ):
        # x is token-IDs shape: (batch, seq_len)
        # (batch_size, seq_len, d_model)
        decoder_input_ids = self.pos_embedding(decoder_input_ids)
        # Add dropout.
        decoder_input_ids = self.dropout(decoder_input_ids)
        # Apply each decoder layer sequentially
        x_key_padding_mask = x_key_padding_mask == 0
        context_key_padding_mask = context_key_padding_mask == 0
        for layer in self.dec_layers:
            decoder_input_ids = layer(
                decoder_input_ids,
                context,
                x_key_padding_mask=x_key_padding_mask,
                context_key_padding_mask=context_key_padding_mask,
            )
        decoder_input_ids = self.layernorm(decoder_input_ids)
        self.last_attn_scores = self.dec_layers[-1].cross_attn.last_attn_scores
        # (batch_size, seq_len, d_model)
        return decoder_input_ids


sample_decoder = Decoder(
    d_model=512, num_heads=4, d_ff=2048, num_layers=6, dropout_rate=0.1
)
sample_batch_decoder_input = next(iter(val_dataloader))
decoder_input_ids = sample_batch_decoder_input["labels"]
decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).int()

context_tensor = torch.randn(
    sample_batch_decoder_input["input_ids"].shape[0],
    sample_batch_decoder_input["input_ids"].shape[1],
    512,
)
context_attention_mask_for_decoder = sample_batch_decoder_input["attention_mask"]

output = sample_decoder(
    decoder_input_ids,
    context_tensor,
    x_key_padding_mask=decoder_attention_mask,
    context_key_padding_mask=context_attention_mask_for_decoder,
)
assert output.shape == (decoder_input_ids.shape[0], decoder_input_ids.shape[1], 512)
print("Decoder forward pass successful with padding mask.")

