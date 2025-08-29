import torch


def train(model, train_iter, src_field, tgt_field, optimizer, criterion):
    model.train()
    total_loss = 0
    for i, data in enumerate(train_iter):
        # The target input is the target sequence without the EOS token.
        # This is what the decoder receives as input.
        src, tgt = data.src, data.tgt
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

        # backpropagation
        optimizer.zero_grad()
        loss = criterion(output, tgt_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            avg_loss_100 = total_loss / (i + 1)
            print(f"  Step: {i+1} | Avg Train Loss: {avg_loss_100:.3f}")

    return total_loss / len(train_iter)
