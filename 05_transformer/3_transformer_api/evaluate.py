import torch


def evaluate(model, valid_dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt, src_key_padding_mask, tgt_key_padding_mask in valid_dataloader:
            # The target input is the target sequence without the EOS token
            tgt_input = tgt[:, :-1]
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]

            src_key_padding_mask = src_key_padding_mask == 0
            tgt_key_padding_mask = tgt_key_padding_mask == 0

            # [batch_size, tgt_seq_len - 1, vocab_size]
            logits = model.forward(
                src, tgt_input, src_key_padding_mask, tgt_key_padding_mask
            )
            output = logits.reshape(-1, logits.shape[-1])
            # [batch_size, tgt_seq_len - 1] -> [batch x (tgt_seq_len - 1)]
            tgt_out = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt_out)
            total_loss += loss.item()

    return total_loss / len(valid_dataloader)
