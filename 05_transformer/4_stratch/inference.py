import torch
from preprocess import (
    text_transform,
    src_language,
    tgt_rev_vocab,
    special_tokens,
)


def greedy_decode(model, src_sentence, device, batch_first, max_len=50):
    model.eval()
    if batch_first:
        src = text_transform[src_language](src_sentence).to(device).unsqueeze(0)
        src_padding_mask = src == special_tokens["<pad>"]
    else:
        src = text_transform[src_language](src_sentence).to(device).unsqueeze(1)
        src_padding_mask = (src == special_tokens["<pad>"]).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(special_tokens["<bos>"]).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            if batch_first:
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    ys.size(1)
                ).to(device)
                tgt_padding_mask = ys == special_tokens["<pad>"]
            else:
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    ys.size(0)
                ).to(device)
                tgt_padding_mask = (ys == special_tokens["<pad>"]).transpose(0, 1)
            out = model.decode(
                ys,
                memory,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

        prob = model.generator(out[:, -1, :] if batch_first else out[-1, :, :])
        next_word_idx = torch.argmax(prob, dim=1).item()

        next_word_tensor = torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)
        if batch_first:
            ys = torch.cat([ys, next_word_tensor], dim=1)
        else:
            ys = torch.cat([ys, next_word_tensor], dim=0)

        if next_word_idx == special_tokens["<eos>"]:
            break

    return ys.flatten()


def topk_decode(model, src_sentence, device, batch_first, max_len=50, k=5):
    """
    Decodes a source sentence using top-k sampling.
    """
    model.eval()

    if batch_first:
        src = text_transform[src_language](src_sentence).to(device).unsqueeze(0)
        src_padding_mask = src == special_tokens["<pad>"]
    else:
        src = text_transform[src_language](src_sentence).to(device).unsqueeze(1)
        src_padding_mask = (src == special_tokens["<pad>"]).transpose(0, 1)

    with torch.no_grad():
        memory = model.encode(src, src_padding_mask)

    ys = torch.ones(1, 1).fill_(special_tokens["<bos>"]).type(torch.long).to(device)

    for _ in range(max_len - 1):
        with torch.no_grad():
            if batch_first:
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    ys.size(1)
                ).to(device)
                tgt_padding_mask = ys == special_tokens["<pad>"]
            else:
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    ys.size(0)
                ).to(device)
                tgt_padding_mask = (ys == special_tokens["<pad>"]).transpose(0, 1)
            out = model.decode(
                ys,
                memory,
                memory_key_padding_mask=src_padding_mask,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

        prob = model.generator(out[:, -1, :] if batch_first else out[-1, :, :])

        # Get the top-k probabilities and indices
        topk_probs, topk_indices = torch.topk(prob, k, dim=1)

        # Sample one from the top-k indices
        next_word_idx = topk_indices.gather(
            1, torch.multinomial(topk_probs, num_samples=1)
        ).item()

        next_word_tensor = torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)
        if batch_first:
            ys = torch.cat([ys, next_word_tensor], dim=1)
        else:
            ys = torch.cat([ys, next_word_tensor], dim=0)

        if next_word_idx == special_tokens["<eos>"]:
            break

    return ys.flatten()


def translate(model, src_sentence, device, decode_fn, batch_first):
    tgt_tokens = decode_fn(model, src_sentence, device, batch_first).cpu().numpy()

    def lookup_tokens(indices):
        return [tgt_rev_vocab.get(i, "<unk>") for i in indices]

    return (
        " ".join(lookup_tokens(list(tgt_tokens)))
        .replace("<bos>", "")
        .replace("<eos>", "")
        .strip()
    )
