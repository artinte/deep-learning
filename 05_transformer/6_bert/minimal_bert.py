import torch
import random

class MiniBERT(torch.nn.Module):
    def __init__(self, vocab_size=30522, hidden=128, n_layers=2, n_heads=2, max_len=128):
        super().__init__()
        self.token_embed = torch.nn.Embedding(vocab_size, hidden)
        self.pos_embed = torch.nn.Embedding(max_len, hidden)
        self.segment_embed = torch.nn.Embedding(2, hidden)

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, dim_feedforward=512, dropout=0.1, activation='gelu')
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = torch.nn.LayerNorm(hidden)

        self.mlm_head = torch.nn.Linear(hidden, vocab_size)
        self.nsp_head = torch.nn.Linear(hidden, 2)

    def forward(self, input_ids, segment_ids):
        B, T = input_ids.size()
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)

        x = self.token_embed(input_ids) + self.pos_embed(positions) + self.segment_embed(segment_ids)
        x = self.norm(x)

        x = x.permute(1, 0, 2)  # Transformer expects (T, B, E)
        encoded = self.encoder(x)
        encoded = encoded.permute(1, 0, 2)  # Back to (B, T, E)

        mlm_logits = self.mlm_head(encoded)
        nsp_logits = self.nsp_head(encoded[:, 0])  # [CLS] token for NSP

        return mlm_logits, nsp_logits

vocab_list = [
    "[PAD]", "[CLS]", "[SEP]", "[MASK]",
    "i", "you", "he", "she", "we", "they", "are", "books",
    "like", "likes", "love", "hate", "enjoy", "prefers",
    "dogs", "cats", "birds", "fish", "pizza", "ice", "cream",
    "run", "eat", "sing", "play", "read", "write", "eats",
    "fast", "slow", "fun", "boring", "good", "bad", "is",
    "loud", "drink", "music"
]
vocab = {word: idx for idx, word in enumerate(vocab_list)}
id2word = {idx: word for word, idx in vocab.items()}
vocab_size = len(vocab)


sentence_pairs = [
    ("i like dogs", "they hate cats", 1),
    ("you enjoy pizza", "we eat ice cream", 1),
    ("he prefers birds", "she likes fish", 1),
    ("i run fast", "they play slow", 1),
    ("they sing loud", "i read books", 1),
    ("dogs are good", "ice cream is bad", 1),
    ("cats are fun", "birds are boring", 1),
    ("we write books", "she eats pizza", 1),
    ("he read books", "she write books", 1),
    ("we eat pizza", "they drink ice cream", 1),
    ("dogs are good", "cats are bad", 1),
    ("i sing loud", "they play music", 1),
    ("i like music", "you like music", 1),
    ("i like fish", "you like ice cream", 1),
    ("i love run", "i hate read", 1),


    ("i like dogs", "books are boring", 0),
    ("we eat ice cream", "he prefers birds", 0),
    ("she likes fish", "they sing loud", 0),
    ("cats are fun", "they hate cats", 0),
    ("she eats pizza", "i run fast", 0),
    ("he prefers birds", "dogs are good", 0),
    ("you enjoy pizza", "we write books", 0),
    ("they play slow", "birds are boring", 0),
    ("we eat pizza", "books are fun", 0),
    ("he write books", "ice cream is good", 0),
    ("they play music", "cats are boring", 0),
    ("she like fish", "i read books", 0)
]


def tokenize(sentence):
    return [vocab.get(w, vocab["[PAD]"]) for w in sentence.split()]

def create_batch(pairs, max_len=16):
    input_ids, token_type_ids, mlm_labels, nsp_labels = [], [], [], []
    for s1, s2, is_next in pairs:
        tokens = ["[CLS]"] + s1.split() + ["[SEP]"] + s2.split() + ["[SEP]"]
        ids = [vocab[t] for t in tokens]
        seg = [0]*(len(s1.split())+2) + [1]*(len(s2.split())+1)

        # Padding
        while len(ids) < max_len:
            ids.append(vocab["[PAD]"])
            seg.append(0)

        # Create MLM target
        mlm_label = [-100] * max_len
        candidate_idxs = [i for i, t in enumerate(ids) if t not in [vocab["[PAD]"], vocab["[CLS]"], vocab["[SEP]"]]]
        if candidate_idxs:
            mask_idx = random.choice(candidate_idxs)
            mlm_label[mask_idx] = ids[mask_idx]
            ids[mask_idx] = vocab["[MASK]"]

        input_ids.append(ids)
        token_type_ids.append(seg)
        mlm_labels.append(mlm_label)
        nsp_labels.append(is_next)

    return (
        torch.tensor(input_ids),
        torch.tensor(token_type_ids),
        torch.tensor(mlm_labels),
        torch.tensor(nsp_labels)
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniBERT(vocab_size=vocab_size, hidden=128, n_layers=2, n_heads=2, max_len=16).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn_mlm = torch.nn.CrossEntropyLoss(ignore_index=-100)
loss_fn_nsp = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(30):
    input_ids, segment_ids, mlm_targets, nsp_targets = create_batch(sentence_pairs)
    input_ids = input_ids.to(device)
    segment_ids = segment_ids.to(device)
    mlm_targets = mlm_targets.to(device)
    nsp_targets = nsp_targets.to(device)

    mlm_logits, nsp_logits = model(input_ids, segment_ids)

    loss_mlm = loss_fn_mlm(mlm_logits.view(-1, vocab_size), mlm_targets.view(-1))
    loss_nsp = loss_fn_nsp(nsp_logits, nsp_targets)

    loss = loss_mlm + loss_nsp
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch:02d} | MLM Loss: {loss_mlm.item():.4f} | NSP Loss: {loss_nsp.item():.4f}")
