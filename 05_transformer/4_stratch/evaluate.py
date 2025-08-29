import torch


def evaluate(model, valid_iter, src_field, tgt_field, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in valid_iter:
            # The target input is the target sequence without the EOS token
            src = data.src
            tgt = data.tgt
            src_key_padding_mask = src == src_field.vocab.stoi[src_field.pad_token]
            tgt_key_padding_mask = tgt == tgt_field.vocab.stoi[tgt_field.pad_token]
            tgt_input = tgt[:, :-1]
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            # [batch_size, tgt_seq_len, vocab_size]
            logits = model.forward(
                src, tgt_input, src_key_padding_mask, tgt_key_padding_mask
            )
            output = logits.reshape(-1, logits.shape[-1])
            # [batch, seq_len] -> [batch x seq_len]
            tgt_out = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()

    return total_loss / len(valid_iter)
