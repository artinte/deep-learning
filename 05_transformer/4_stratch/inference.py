import torch


def translate_one_batch(model, iterator, src_field, tgt_field, max_len=50):
    model.eval()
    translations = []

    with torch.no_grad():
        batch = next(iter(iterator))
        src = batch.src
        tgt = batch.tgt
        device = src.device

        # 1. Encode the source sequence once
        memory = model.encode(src)

        # 2. Prepare the decoder's initial input with <sos> tokens
        batch_size = src.size(0)
        tgt_tokens = torch.full(
            (batch_size, 1),
            tgt_field.vocab.stoi[tgt_field.init_token],
            dtype=torch.long,
            device=device,
        )

        # 3. Autoregressively generate the translation
        for i in range(max_len):
            # Decode the next token based on the generated sequence so far
            output = model.decode(tgt_tokens, memory)

            # Get the predicted token (highest probability from the last time step)
            next_token_preds = output[:, -1, :].argmax(dim=-1)

            # Append the predicted token to the sequence
            tgt_tokens = torch.cat([tgt_tokens, next_token_preds.unsqueeze(1)], dim=-1)

            # Break if all sequences in the batch have generated <eos>
            if (next_token_preds == tgt_field.vocab.stoi[tgt_field.eos_token]).all():
                break

        # 4. Post-process and collect results
        # The generated tokens now have shape [batch_size, generated_len]
        for i in range(batch_size):
            src_tokens = [src_field.vocab.itos[idx] for idx in src[i]]
            hyp_tokens = []

            # Skip the <sos> token at the start of the generated sequence
            for idx in tgt_tokens[i][1:]:
                if idx == tgt_field.vocab.stoi[tgt_field.eos_token]:
                    break
                if idx != tgt_field.vocab.stoi[tgt_field.pad_token]:
                    hyp_tokens.append(tgt_field.vocab.itos[idx])

            # Get the original target for comparison (optional but good practice)
            tgt_orig_tokens = [
                tgt_field.vocab.itos[idx]
                for idx in tgt[i][1:]
                if idx != tgt_field.vocab.stoi[tgt_field.pad_token]
            ]

            translations.append(
                {
                    "src": " ".join(src_tokens),
                    "hyp": " ".join(hyp_tokens),
                    "tgt": " ".join(tgt_orig_tokens),
                }
            )

    return translations
