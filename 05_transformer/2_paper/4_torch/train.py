import torch

def train(model, train_dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in train_dataloader:
        src = src.to(model.device)
        tgt = tgt.to(model.device)
        
        # The target input is the target sequence without the EOS token
        tgt_input = tgt[:, :-1]
        # [batch_size, tgt_seq_len, vocab_size]
        logits = model.forward(src, tgt_input)
        output = logits.reshape(-1, logits.shape[-1])

        # [batch, seq_len] -> [batch x seq_len]
        tgt_out = tgt[:, 1:].reshape(-1)
        
        # backpropagation
        optimizer.zero_grad()
        loss = criterion(output, tgt_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)
