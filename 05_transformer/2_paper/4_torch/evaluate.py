import torch


def evaluate(model, valid_dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in valid_dataloader:
            src = src.to(model.device)
            tgt = tgt.to(model.device)

            # The target input is the target sequence without the EOS token
            tgt_input = tgt[:, :-1]
            # [batch_size, tgt_seq_len, vocab_size]
            logits = model.forward(src, tgt_input)
            output = logits.reshape(-1, logits.shape[-1])

            # [batch, seq_len] -> [batch x seq_len]
            tgt_out = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_out)

            total_loss += loss.item()

    return total_loss / len(valid_dataloader)
