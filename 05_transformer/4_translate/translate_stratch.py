import torch
import pathlib
import sys
import nltk
import torch
import copy

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from common import MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, \
    Encoder, EncoderLayer, Decoder, DecoderLayer, Embeddings, Generator

import datasets
from torchtext.data import Field, Example, Dataset, BucketIterator
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('punkt_tab')

dataset = datasets.load_dataset('bentrevett/multi30k')
print(dataset)
# {'en': 'Two young, White males are outside near many bushes.',
# 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
print(dataset['train'][0])

train_data = [(example['de'], example['en']) for example in dataset['train']]
valid_data = [(example['de'], example['en']) for example in dataset['validation']]
test_data = [(example['de'], example['en']) for example in dataset['test']]

SRC = Field(tokenize=word_tokenize, init_token='<sos>', eos_token='<eos>', pad_token='<pad>', lower=True, batch_first=True)
TRG = Field(tokenize=word_tokenize, init_token='<sos>', eos_token='<eos>', pad_token='<pad>', lower=True, batch_first=True)

train_examples = [Example.fromlist([src, trg], fields=[('src', SRC), ('trg', TRG)]) for src, trg in train_data]
valid_examples = [Example.fromlist([src, trg], fields=[('src', SRC), ('trg', TRG)]) for src, trg in valid_data]
test_examples = [Example.fromlist([src, trg], fields=[('src', SRC), ('trg', TRG)]) for src, trg in test_data]

train_dataset = Dataset(examples=train_examples, fields=[('src', SRC), ('trg', TRG)])
valid_dataset = Dataset(examples=valid_examples, fields=[('src', SRC), ('trg', TRG)])
test_dataset = Dataset(examples=test_examples, fields=[('src', SRC), ('trg', TRG)])

SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

print('Source vocabulary size: ' + str(len(SRC.vocab)))
print('Target vocabulary size: ' + str(len(TRG.vocab)))

print([word for word, _ in list(SRC.vocab.stoi.items())[:10]])
print([word for word, _ in list(TRG.vocab.stoi.items())[:10]])

src_vocab_size = len(SRC.vocab)
trg_vocab_size = len(TRG.vocab)

BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_dataset, valid_dataset, test_dataset),
        batch_size=BATCH_SIZE,
        device=device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
    )


BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
)

for batch in train_iterator:
    src = batch.src
    trg = batch.trg
    
    print(src.shape)
    print(trg.shape)
    break


class EncoderDecoder(torch.nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt):
        """
        Take in and process masked src and target sequences.
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    
    def make_src_mask(self, src):
        # # shape: (batch_size, 1, 1, src_len)
        return (src != TRG.vocab.stoi[TRG.pad_token]).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        # Create look-ahead mask (causal mask) + padding mask
        batch_size, tgt_len = tgt.size()
        
        # Look-ahead mask
        subsequent_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
        
        # Padding mask
        padding_mask = (tgt != TRG.vocab.stoi[TRG.pad_token]).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_len)

        # Combine: pad_mask & look-ahead
        combined_mask = padding_mask & ~subsequent_mask.unsqueeze(0)  # (1, tgt_len, tgt_len) broadcasted
        return combined_mask


class TransformerModel(torch.nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 emb_size=256, nhead=8, nhid=1024, nlayers=6, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.src_emb = torch.nn.Embedding(src_vocab_size, emb_size)
        self.trg_emb = torch.nn.Embedding(trg_vocab_size, emb_size)
        
        self.transformer = torch.nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = torch.nn.Linear(emb_size, trg_vocab_size)
    
    def forward(self, src, trg):
        src_emb = self.src_emb(src)
        trg_emb = self.trg_emb(trg)
        
        output = self.transformer(src_emb, trg_emb)
        
        return self.fc_out(output)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # Helper: Construct a model from hyperparameters.
    # c = copy.deepcopy
    # attn = MultiHeadedAttention(h, d_model)
    # ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # position = PositionalEncoding(d_model, dropout)
    # model = EncoderDecoder(
    #     Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
    #     Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
    #     torch.nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
    #     torch.nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
    #     Generator(d_model, tgt_vocab),
    # )
    
    # # This was important from their code.
    # # Initialize parameters with Glorot / fan_avg.
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         torch.nn.init.xavier_uniform_(p)
    model = TransformerModel(src_vocab_size, trg_vocab_size)
    return model

model = make_model(src_vocab_size, trg_vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

criterion.to(device)

EPOCHS = 40
CLIP = 1.0  # Gradient clipping

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    
    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        assert src.size(0) == trg.size(0), f"src batch size {src.size(0)} does not match trg batch size {trg.size(0)}"
        
        optimizer.zero_grad()
        
        # Exclude the last token from target sequence (shifted target sequence)
        output = model(src, trg[:, :-1])
        
        output_dim = output.shape[-1]
        
        # Flatten output and trg to calculate loss
        output = output.view(-1, output_dim)  # (batch_size * seq_len, trg_vocab_size)
        trg = trg[:, 1:].contiguous().view(-1)  # Exclude first token from target sequence
        
        loss = criterion(output, trg)
        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval() 
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            
            output = model(src, trg[:, :-1])
            
            output_dim = output.shape[-1]
            
            output = output.view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

for epoch in range(EPOCHS):
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}")

def translate_one_batch(model, iterator, SRC, TRG):
    model.eval()
    translations = []
    
    with torch.no_grad():
        batch = next(iter(iterator))  
        src = batch.src
        trg = batch.trg
        
        output = model(src, trg[:, :-1])
        output = output.argmax(dim=-1)
        
        for i in range(src.size(0)):
            src_tokens = [SRC.vocab.itos[idx] for idx in src[i]]
            hyp_tokens = []
            for idx in output[i]:
                if idx == TRG.vocab.stoi[TRG.eos_token]:
                    break
                if idx != TRG.vocab.stoi[TRG.pad_token]:
                    hyp_tokens.append(TRG.vocab.itos[idx])

            translations.append({
                'src': ' '.join(src_tokens),
                'hyp': ' '.join(hyp_tokens),
                'trg': ' '.join([TRG.vocab.itos[idx] for idx in trg[i][1:].cpu().numpy() if idx != TRG.vocab.stoi[TRG.pad_token]])
            })
    
    return translations

def print_translations(translations):
    for translation in translations:
        print(f"Source: {translation['src']}")
        print(f"Prediction: {translation['hyp']}")
        print(f"Reference: {translation['trg']}")
        print("-" * 50)

translations = translate_one_batch(model, test_iterator, SRC, TRG)
print_translations(translations)
