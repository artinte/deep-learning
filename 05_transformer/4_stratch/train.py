import torch


def train(model, train_dataloader, optimizer, criterion):
    """
    Trains the Transformer model.
    Args:
        model (torch.nn.Module): The Transformer model.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (torch.nn.modules.loss._Loss): The loss function.
    """
    model.train()
    total_loss = 0
    for src, tgt, src_key_padding_mask, tgt_key_padding_mask in train_dataloader:
        # The target input is the target sequence without the EOS token.
        # This is what the decoder receives as input.
        tgt_input = tgt[:, :-1]
        tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

        src_key_padding_mask = src_key_padding_mask == 0
        tgt_key_padding_mask = tgt_key_padding_mask == 0

        # [batch_size, tgt_seq_len, vocab_size]
        logits = model.forward(
            src, tgt_input, src_key_padding_mask, tgt_key_padding_mask
        )
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
