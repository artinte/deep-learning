import torch
from preprocess import special_tokens
from utils import create_mask


def evaluate(model, dataloader, loss_fn, device, batch_first):
    model.eval()
    losses = 0
    for src, tgt in dataloader:
        if batch_first:
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
        else:
            tgt_input = tgt[:-1, :]
            tgt_out = tgt[1:, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input, special_tokens["<pad>"], device, batch_first
        )

        with torch.no_grad():
            logits = model(
                src,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
            )

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(dataloader)
