import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # Embedding layer: maps input tokens to embedding vectors
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Bidirectional LSTM layer
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [src_len, batch_size]
        
        embedded = self.dropout(self.embedding(src))
        # embedded = [src_len, batch_size, emb_dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src_len, batch_size, hid_dim * 2]
        # hidden = [n_layers * 2, batch_size, hid_dim]
        # cell = [n_layers * 2, batch_size, hid_dim]
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        cell = torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)
        # hidden = [batch_size, hid_dim * 2]
        # cell = [batch_size, hid_dim * 2]
        
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        self.hid_dim = hid_dim
        
        # Bahdanau attention mechanism as proposed in the paper
        self.attn = nn.Linear(hid_dim * 3, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [batch_size, hid_dim * 2]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # Repeat hidden state to match the length of encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # hidden = [batch_size, src_len, hid_dim * 2]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, hid_dim * 2]
        
        # Calculate attention energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
        # energy = [batch_size, src_len, hid_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention = [batch_size, src_len]
        
        return F.softmax(attention, dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        
        assert hid_dim % n_heads == 0, "Hidden dimension must be divisible by number of heads"
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        # Linear layers for query, key, value projections
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        # Output projection layer
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor as proposed in the paper
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
        
    def forward(self, query, key, value, mask = None):
        # query = [batch_size, trg_len, hid_dim]
        # key = [batch_size, src_len, hid_dim]
        # value = [batch_size, src_len, hid_dim]
        
        batch_size = query.shape[0]
        
        # Project query, key, value
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)
        # q, k, v = [batch_size, seq_len, hid_dim]
        
        # Reshape and split into multiple heads
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        # q = [batch_size, n_heads, trg_len, head_dim]
        # k = [batch_size, n_heads, src_len, head_dim]
        # v = [batch_size, n_heads, src_len, head_dim]
        
        # Calculate scaled dot-product attention
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch_size, n_heads, trg_len, src_len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(energy, dim = -1)
        # attention = [batch_size, n_heads, trg_len, src_len]
        
        x = torch.matmul(self.dropout(attention), v)
        # x = [batch_size, n_heads, trg_len, head_dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch_size, trg_len, n_heads, head_dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch_size, trg_len, hid_dim]
        
        x = self.fc_o(x)
        # x = [batch_size, trg_len, hid_dim]
        
        return x, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, n_heads, dropout, attention):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        # Embedding layer for target language
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # LSTM layer that takes both embedding and context vector
        self.rnn = nn.LSTM(emb_dim + hid_dim * 2, hid_dim * 2, n_layers, dropout=dropout)
        
        # Final linear layer to produce output probabilities
        self.fc_out = nn.Linear(hid_dim * 4 + emb_dim, output_dim)
        
        self.multi_head_attention = MultiHeadAttention(hid_dim * 2, n_heads, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input = [batch_size]
        # hidden = [batch_size, hid_dim * 2]
        # cell = [batch_size, hid_dim * 2]
        # encoder_outputs = [src_len, batch_size, hid_dim * 2]
        
        input = input.unsqueeze(0)
        # input = [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        # Calculate attention weights
        attn = self.attention(hidden, encoder_outputs)
        # attn = [batch_size, src_len]
        
        attn = attn.unsqueeze(1)
        # attn = [batch_size, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch_size, src_len, hid_dim * 2]
        
        # Calculate context vector
        context = torch.bmm(attn, encoder_outputs)
        # context = [batch_size, 1, hid_dim * 2]
        
        context = context.permute(1, 0, 2)
        # context = [1, batch_size, hid_dim * 2]
        
        # Concatenate embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim = 2)
        # rnn_input = [1, batch_size, emb_dim + hid_dim * 2]
        
        output, (hidden, cell) = self.rnn(rnn_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        # output = [1, batch_size, hid_dim * 2]
        # hidden = [1, batch_size, hid_dim * 2]
        # cell = [1, batch_size, hid_dim * 2]
        
        # Apply multi-head attention
        output = output.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(0, 1, 2)
        multi_head_out, _ = self.multi_head_attention(output, encoder_outputs, encoder_outputs)
        multi_head_out = multi_head_out.permute(1, 0, 2)
        
        # Prepare tensors for final prediction
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        context = context.squeeze(0)
        multi_head_out = multi_head_out.squeeze(0)
        
        # Concatenation as proposed in the paper
        prediction = self.fc_out(torch.cat((output, context, embedded, multi_head_out), dim = 1))
        # prediction = [batch_size, output_dim]
        
        return prediction, hidden.squeeze(0), cell.squeeze(0), attn.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_forcing_ratio: probability of using teacher forcing
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # Initialize tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # Get encoder outputs
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # First input to the decoder is the <sos> token
        input = trg[0,:]
        
        for t in range(1, trg_len):
            # Forward pass through decoder
            output, hidden, cell, attn = self.decoder(input, hidden, cell, encoder_outputs)
            
            # Store decoder output
            outputs[t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get predicted token
            top1 = output.argmax(1)
            
            # Update input for next time step
            input = trg[t] if teacher_force else top1
            
        return outputs

# Model initialization example
def init_model(input_dim, output_dim, device):
    # Hyperparameters
    emb_dim = 256
    hid_dim = 512
    n_layers = 2
    n_heads = 8
    dropout = 0.5
    
    # Initialize components
    attn = Attention(hid_dim)
    encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
    decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, n_heads, dropout, attn)
    
    # Initialize sequence-to-sequence model
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    return model

# Training function
def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src = batch.src.to(device)
        trg = batch.trg.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        # output shape: [trg_len, batch_size, output_dim]
        # target shape: [trg_len, batch_size]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion, device):
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src.to(device)
            trg = batch.trg.to(device)
            
            output = model(src, trg, 0)  # No teacher forcing during evaluation
            
            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# Main function example
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Assume vocabulary sizes (adjust based on actual dataset)
    src_vocab_size = 10000
    trg_vocab_size = 10000
    
    # Initialize model
    model = init_model(src_vocab_size, trg_vocab_size, device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    
    # Here you should load your data iterators
    # train_iterator, valid_iterator, test_iterator = ...
    
    # Training parameters
    N_EPOCHS = 10
    CLIP = 1
    
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        print(f'Epoch {epoch+1}/{N_EPOCHS}')
        
        # Train the model
        # train_loss = train(model, train_iterator, optimizer, criterion, CLIP, device)
        # valid_loss = evaluate(model, valid_iterator, criterion, device)
        
        # Example values for demonstration
        train_loss = 3.0 - epoch * 0.2
        valid_loss = 3.2 - epoch * 0.15
        
        print(f'Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'attention_nmt_model.pt')
    
    # Load best model and evaluate on test set
    # model.load_state_dict(torch.load('attention_nmt_model.pt'))
    # test_loss = evaluate(model, test_iterator, criterion, device)
    # print(f'Test Loss: {test_loss:.3f}')

if __name__ == '__main__':
    main()
