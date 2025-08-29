import torch


def greedy_translate(model, sentence, src_field, tgt_field, max_len=512):

    model.eval()
    with torch.no_grad():
        tokens = [token for token in src_field.tokenizer(sentence)]
        src_tokens = [src_field.vocab.stoi[token] for token in tokens]
        # [1, src_seq_len]
        src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(model.device)
        src_key_padding_mask = src_tensor == src_field.vocab.stoi[src_field.pad_token]

        memory = model.encode(src_tensor, src_key_padding_mask)

        bos_token_id = tgt_field.vocab.stoi[tgt_field.bos_token]
        tgt_tokens = [bos_token_id]

        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(model.device)
            # The decoder's memory padding mask is the source padding mask.
            logits = model.decode(tgt_tensor, memory, src_key_padding_mask)
            # Greedily select the next token.
            next_token_id = logits.argmax(dim=-1)[0, -1].item()
            tgt_tokens.append(next_token_id)

            if next_token_id == tgt_field.vocab.stoi[tgt_field.eos_token]:
                break

    translation = " ".join(
        [
            tgt_field.vocab.itos[token]
            for token in tgt_tokens
            if token not in [bos_token_id, tgt_field.vocab.stoi[tgt_field.eos_token]]
        ]
    )
    return translation
