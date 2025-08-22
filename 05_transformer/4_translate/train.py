import torch


def train(model, iterator, optimizer, criterion, clip=1.0):
    model.train()

    epoch_loss = 0
    for _, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        assert src.size(0) == trg.size(
            0
        ), f"src batch size {src.size(0)} does not match trg batch size {trg.size(0)}"

        optimizer.zero_grad()

        # Exclude the last token from target sequence (shifted target sequence)
        output = model(src, trg[:, :-1])

        output_dim = output.shape[-1]
        # Flatten output and trg to calculate loss
        # (batch_size * seq_len, trg_vocab_size)
        output = output.view(-1, output_dim)
        # Exclude first token from target sequence
        trg = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
