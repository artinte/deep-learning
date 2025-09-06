import torch


class Config:
    batch_size = 64  # How many independent sequences will we process in parallel.
    block_size = 2  # What is the maxium context length for predictions
    max_iters = 5000  # How many training iterations
    eval_interval = 500  # How often to evaluate the model
    learning_rate = 3e-4  # Learning rate for the optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
    eval_iters = 200  # How many batches to use for evaluation
    n_embd = 384  # The embedding dimension
    n_head = 6  # Number of attention heads
    n_layer = 6  # Number of layers in the Transformer
    dropout = 0.0  # Dropout rate for regularization


torch.manual_seed(1337)

text = """
床前明月光，疑是地上霜。
举头望明月，低头思故乡。
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:", vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i+Config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+Config.block_size+1] for i in ix])
    x, y = x.to(Config.device), y.to(Config.device)
    return x, y


# The main GPT language model
class GPTLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding_table = torch.nn.Embedding(
            Config.block_size, Config.n_embd
        )

        # Use a single TransformerEncoderLayer to define the block
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=Config.n_embd,
            nhead=Config.n_head,
            dim_feedforward=4 * Config.n_embd,
            dropout=Config.dropout,
            batch_first=True,
        )

        # Stack multiple layers using TransformerEncoder
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=Config.n_layer
        )

        self.ln_f = torch.nn.LayerNorm(Config.n_embd)
        self.lm_head = torch.nn.Linear(Config.n_embd, vocab_size)

        # We pre-compute the causal mask for efficiency.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(Config.block_size, Config.block_size) * float("-inf"),
                diagonal=1,
            ),
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=Config.device)
        )  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) - Add token and positional embeddings

        # The attention mask needs to be the same size as the block_size
        attn_mask = self.causal_mask[:T, :T]

        # The TransformerEncoder expects a mask with shape (T, T)
        x = self.transformer_encoder(x, mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -Config.block_size :]
            # Get predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# --- Training and Inference ---
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(Config.eval_iters)
        for k in range(Config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = GPTLanguageModel().to(Config.device)
print(sum(p.numel() for p in model.parameters()) / 1e6, "Model Parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
for iter in range(Config.max_iters):
    if iter % Config.eval_interval == 0:
        losses = estimate_loss(model)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

start_seed = "床前明月光"
context = torch.tensor(
    encode(start_seed), dtype=torch.long, device=Config.device
).unsqueeze(0)
generated_text = model.generate(context, max_new_tokens=500)[0]
print(decode(generated_text.tolist()))
