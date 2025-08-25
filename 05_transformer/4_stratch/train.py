import torch


def train(model, iterator, optimizer, criterion, clip=1.0):
    model.train()

    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.tgt

        optimizer.zero_grad()

        # Exclude the last token from target sequence (shifted target sequence)
        output = model(src, tgt[:, :-1])

        output_dim = output.shape[-1]
        # Flatten output and tgt to calculate loss
        # (batch_size * seq_len, tgt_vocab_size)
        output = output.view(-1, output_dim)
        # Exclude first token from target sequence
        tgt = tgt[:, 1:].contiguous().view(-1)
        loss = criterion(output, tgt)
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
