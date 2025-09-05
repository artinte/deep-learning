import torch
from utils import create_mask
from preprocess import special_tokens


def train(model, optimizer, dataloader, loss_fn, device, batch_first):
    model.train()
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

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
        )

        optimizer.zero_grad()
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        # Add the gradient clipping for stable training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
        optimizer.step()
        losses += loss.item()
    return losses / len(dataloader)
