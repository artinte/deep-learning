import torch


def translate_one_batch(model, iterator, SRC, TRG):
    model.eval()
    translations = []

    with torch.no_grad():
        batch = next(iter(iterator))
        src = batch.src
        trg = batch.trg

        output = model(src, trg[:, :-1])
        output = output.argmax(dim=-1)

        for i in range(src.size(0)):
            src_tokens = [SRC.vocab.itos[idx] for idx in src[i]]
            hyp_tokens = []
            for idx in output[i]:
                if idx == TRG.vocab.stoi[TRG.eos_token]:
                    break
                if idx != TRG.vocab.stoi[TRG.pad_token]:
                    hyp_tokens.append(TRG.vocab.itos[idx])

            translations.append(
                {
                    "src": " ".join(src_tokens),
                    "hyp": " ".join(hyp_tokens),
                    "trg": " ".join(
                        [
                            TRG.vocab.itos[idx]
                            for idx in trg[i][1:].cpu().numpy()
                            if idx != TRG.vocab.stoi[TRG.pad_token]
                        ]
                    ),
                }
            )

    return translations
