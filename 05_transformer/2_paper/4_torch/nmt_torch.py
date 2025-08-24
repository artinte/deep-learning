import transformers
import torch
import math
import datasets

torch.set_printoptions(profile="full")

data_train, data_valid, data_test = datasets.load_dataset(
    "bentrevett/multi30k", split=["train", "validation", "test"]
)
print(f"Training dataset length: {len(data_train)}")
print(f"Validation dataset length: {len(data_valid)}")
print(f"Test dataset length: {len(data_test)}")
print(f"First train sample: {data_train[0]}")

tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")

print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"Unknown token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
print(f"Max truncation length: {tokenizer.model_max_length}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        en_text = self.dataset[index]["en"]
        de_text = self.dataset[index]["de"]
        return en_text, de_text


train_dataset = TranslationDataset(data_train)
valid_dataset = TranslationDataset(data_valid)
test_dataset = TranslationDataset(data_test)


def collate_fn(batch):
    en_sentences = [item[0] for item in batch]
    de_sentences = [item[1] for item in batch]

    src_tokens = tokenizer(
        en_sentences, truncation=True, padding=True, return_tensors="pt"
    )
    tgt_tokens = tokenizer(
        de_sentences, truncation=True, padding=True, return_tensors="pt"
    )

    # The tokenizer now returns a dictionary with 'input_ids' and 'attention_mask'
    # We only need the input IDs for this model.
    return src_tokens["input_ids"], tgt_tokens["input_ids"]


BATCH_SIZE = 16

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

test_src_sample, test_tgt_sample = next(iter(test_dataloader))
print(f"Shape of test src sample: {test_src_sample.shape}")
print(f"First batch test src token: {test_src_sample}")
print(f"Shape of test tgt sample: {test_tgt_sample.shape}")
print(f"First batch test tgt token: {test_tgt_sample}")


class PositionalEncoding(torch.nn.Module):
    """
    Injects positional information into the embeddings.
    """

    def __init__(self, d_model, dropout=0.1, max_len=8192):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Seq2SeqTransformer(torch.nn.Module):
    """
    A full sequence-to-sequence model using torch.nn.Transformer.
    This version uses batch_first=True for more intuitive tensor handling.
    """

    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        d_model,
        n_head,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout=0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()

        self.src_embedding = torch.nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = torch.nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Set to True for batch-first tensors
        )

        self.generator = torch.nn.Linear(d_model, tgt_vocab_size)

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask,
        src_padding_mask,
        tgt_padding_mask,
        memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_embedding(src))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask):
        src_emb = self.positional_encoding(self.src_embedding(src))
        return self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_padding_mask, memory_key_padding_mask):
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt))
        return self.transformer.decoder(
            tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask
        )


# --- Helper Functions for Masks and Training ---

def generate_square_subsequent_mask(sz):
    """
    Generates a causal mask. The mask needs to be square for batch_first.
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    """
    Creates the necessary masks for the Transformer model.
    The padding masks are now derived directly from the batch-first tensors.
    """
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool).to(device)

    src_padding_mask = src == tokenizer.pad_token_id
    tgt_padding_mask = tgt == tokenizer.pad_token_id
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # The target input is the target sequence without the EOS token
        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        # The output and target for loss calculation
        output_dim = logits.shape[-1]
        output = logits.reshape(-1, output_dim)
        tgt_out = tgt[:, 1:].reshape(-1)

        # Backpropagation
        optimizer.zero_grad()
        loss = criterion(output, tgt_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def evaluate(model, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in valid_dataloader:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                src, tgt_input
            )

            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            output_dim = logits.shape[-1]
            output = logits.reshape(-1, output_dim)
            tgt_out = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()

    return total_loss / len(valid_dataloader)


SRC_VOCAB_SIZE = tokenizer.vocab_size
TGT_VOCAB_SIZE = tokenizer.vocab_size
D_MODEL = 512
N_HEAD = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.3
NUM_EPOCHS = 15

model = Seq2SeqTransformer(
    num_encoder_layers=NUM_ENCODER_LAYERS,
    num_decoder_layers=NUM_DECODER_LAYERS,
    d_model=D_MODEL,
    n_head=N_HEAD,
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
).to(device)

criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

print("Starting model training...")
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, optimizer, criterion)
    valid_loss = evaluate(model, criterion)
    print(
        f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}"
    )


def translate(model, sentence, max_len=tokenizer.model_max_length):
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer(
            sentence, truncation=True, padding=False, return_tensors="pt"
        )
        src_tensor = src_tokens["input_ids"].to(device)

        src_mask = None
        src_padding_mask = src_tensor == tokenizer.pad_token_id

        memory = model.encode(src_tensor, src_mask, src_padding_mask)
        tgt_tokens = [tokenizer.eos_token_id]

        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(device)

            tgt_mask = (generate_square_subsequent_mask(tgt_tensor.shape[1])).to(device)

            # The `decode` function needs the causal mask, the memory (encoder output), and the memory padding mask.
            logits = model.decode(tgt_tensor, memory, tgt_mask, None, src_padding_mask)

            next_token_id = logits.argmax(dim=-1)[0, -1].item()
            tgt_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return translation


print("Testing Translation on First 32 Samples")
for i in range(32):
    en_sentence = data_test[i]["en"]
    de_reference = data_test[i]["de"]

    translated = translate(model, en_sentence)

    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {translated}")
    print(f"Reference: {de_reference}")
