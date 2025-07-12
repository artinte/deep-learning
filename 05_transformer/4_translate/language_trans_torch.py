import nltk
import torch
from torch import nn, optim
import datasets
from torchtext.data import Field, Example, Dataset, BucketIterator
from nltk.tokenize import word_tokenize

class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 emb_size=256, nhead=8, nhid=1024, nlayers=6, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.src_emb = nn.Embedding(src_vocab_size, emb_size)
        self.trg_emb = nn.Embedding(trg_vocab_size, emb_size)
        
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=nlayers,
            num_decoder_layers=nlayers,
            dim_feedforward=nhid,
            batch_first=True,
            dropout=dropout,
        )

        self.fc_out = nn.Linear(emb_size, trg_vocab_size)
    
    def forward(self, src, trg):
        src_emb = self.src_emb(src)
        trg_emb = self.trg_emb(trg)
        
        output = self.transformer(src_emb, trg_emb)
        
        return self.fc_out(output)

if __name__ == '__main__':
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

    for batch in train_iterator:
        src = batch.src
        trg = batch.trg
        
        print(src.shape)
        print(trg.shape)
        break
    
    model = TransformerModel(src_vocab_size, trg_vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG.vocab.stoi[TRG.pad_token])

    criterion.to(device)

    EPOCHS = 40
    CLIP = 1.0  # Gradient clipping
    
    def train(model, iterator, optimizer, criterion, clip):
        model.train()
        
        epoch_loss = 0
        for i, batch in enumerate(iterator):
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
                    'trg': ' '.join([TRG.vocab.itos[idx] for idx in trg[i][1:].cpu().numpy() if idx != TRG.vocab.stoi[TRG.pad_token]])  # 目标句子
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
 