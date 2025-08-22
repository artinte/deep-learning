import torch


def translate_one_batch(model, iterator, src_field, trg_field):
    model.eval()
    translations = []

    with torch.no_grad():
        batch = next(iter(iterator))
        src = batch.src
        trg = batch.trg

        output = model(src, trg[:, :-1])
        output = output.argmax(dim=-1)

        for i in range(src.size(0)):
            src_tokens = [src_field.vocab.itos[idx] for idx in src[i]]
            hyp_tokens = []
            for idx in output[i]:
                if idx == trg_field.vocab.stoi[trg_field.eos_token]:
                    break
                if idx != trg_field.vocab.stoi[trg_field.pad_token]:
                    hyp_tokens.append(trg_field.vocab.itos[idx])

            translations.append(
                {
                    "src": " ".join(src_tokens),
                    "hyp": " ".join(hyp_tokens),
                    "trg": " ".join(
                        [
                            trg_field.vocab.itos[idx]
                            for idx in trg[i][1:].cpu().numpy()
                            if idx != trg_field.vocab.stoi[trg_field.pad_token]
                        ]
                    ),
                }
            )

    return translations
