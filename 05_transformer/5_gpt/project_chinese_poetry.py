import torch
from preprocess_chinese_poetry import proprocess, special_tokens


class PoetryDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_poems):
        self.data = encoded_poems

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


class Config:
    batch_size = 128  # How many independent sequences will we process in parallel.
    block_size = 128  # What is the maxium context length for predictions
    epochs = 25000  # How many training iterations
    eval_interval = 500  # How often to evaluate the model
    learning_rate = 3e-4  # Learning rate for the optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    eval_iters = 200  # How many batches to use for evaluation
    n_embd = 384  # The embedding dimension
    n_head = 6  # Number of attention heads
    n_layer = 6  # Number of layers in the Transformer
    dropout = 0.0  # Dropout rate for regularization


# The main GPT language model
class GPTLanguageModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding_table = torch.nn.Embedding(
            Config.block_size, Config.n_embd
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=Config.n_embd,
            nhead=Config.n_head,
            dim_feedforward=4 * Config.n_embd,
            dropout=Config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=Config.n_layer
        )
        self.ln_f = torch.nn.LayerNorm(Config.n_embd)
        self.lm_head = torch.nn.Linear(Config.n_embd, vocab_size)

        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(Config.block_size, Config.block_size) * float("-inf"),
                diagonal=1,
            ),
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=Config.device))
        x = tok_emb + pos_emb
        attn_mask = self.causal_mask[:T, :T]
        x = self.transformer_encoder(x, mask=attn_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            # Add the stopping condition.
            if idx_next.item() == special_tokens["<eos>"]:
                break
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
def collate_fn(batch):
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=special_tokens["<pad>"])
    x = padded_batch[:, :-1]
    y = padded_batch[:, 1:]
    return x, y


torch.manual_seed(1337)  # for reproducibility
encoded_poems, vocab, idx_to_word = proprocess(file_path="data/chinese_poetry.csv")

vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

n = int(0.98 * len(encoded_poems))
print(f"Train/val split: {n}/{len(encoded_poems)-n}")
train_data = encoded_poems[:n]
val_data = encoded_poems[n:]
train_dataset = PoetryDataset(train_data)
val_dataset = PoetryDataset(val_data)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn
)


def get_batch(split_loader):
    xb, yb = next(split_loader)
    xb, yb = xb.to(Config.device), yb.to(Config.device)
    return xb, yb


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader):
    out = {}
    model.eval()
    
    # Create fresh iterators for evaluation
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    
    for split in ["train", "val"]:
        losses = torch.zeros(Config.eval_iters)
        current_iter = train_iter if split == "train" else val_iter
        for k in range(Config.eval_iters):
            try:
                X, Y = get_batch(current_iter)
            except StopIteration:
                # If an iterator runs out of data during evaluation,
                # just break the loop. This can happen with small datasets.
                break
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        
        # Calculate mean loss for the batches that were processed
        out[split] = losses.mean()
        
    model.train()
    return out


model = GPTLanguageModel(vocab_size).to(Config.device)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

train_iter = iter(train_loader)
val_iter = iter(val_loader)

for iter_num in range(Config.epochs):
    if iter_num % len(train_loader) == 0:
        train_iter = iter(train_loader)

    if iter_num % Config.eval_interval == 0:
        losses = estimate_loss(model, train_loader, val_loader)
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # Get a batch from the current training iterator
    xb, yb = get_batch(train_iter)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Inference/Generation
print("-" * 50)
start_seed = "白日依山尽，"
encoded_seed = [vocab.get(char, special_tokens["<unk>"]) for char in start_seed]
context = torch.tensor(encoded_seed, dtype=torch.long, device=Config.device).unsqueeze(
    0
)
generated_ids = model.generate(context, max_new_tokens=200)[0].tolist()

generated_text = "".join([idx_to_word[idx] for idx in generated_ids])
print(f"Generated Poetry: {generated_text}")
