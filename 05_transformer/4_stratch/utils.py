import torch


def create_mask(src, tgt, pad_token_id, device, batch_first):
    if batch_first:
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
    else:
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

    tgt_mask = (
        torch.nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
        .to(device)
        .bool()
    )
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    if batch_first:
        src_padding_mask = src == pad_token_id
        tgt_padding_mask = tgt == pad_token_id
    else:
        src_padding_mask = (src == pad_token_id).transpose(0, 1)
        tgt_padding_mask = (tgt == pad_token_id).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
