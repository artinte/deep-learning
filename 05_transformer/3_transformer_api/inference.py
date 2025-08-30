import torch


def greedy_translate(model, sentence, tokenizer, max_len=512):
    """
    Performs machine translation using a simple greedy search algorithm.

    This method selects the most likely token at each step of the decoding process.
    While fast, it can often lead to suboptimal translations as it does not
    explore alternative, potentially better, sequences. This is typically used
    for quick prototyping or when computational efficiency is the top priority.

    Args:
        model (nn.Module): The trained Transformer translation model.
        sentence (str): The source sentence to translate.
        tokenizer: The tokenizer used for encoding/decoding text.
        max_len (int): The maximum length of the output sequence.

    Returns:
        str: The translated sentence.
    """
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer(
            [sentence], truncation=True, padding=True, return_tensors="pt"
        ).to(model.device)
        src = src_tokens["input_ids"]
        src_key_padding_mask = (src_tokens["attention_mask"] == 0)
        memory = model.encode(src, src_key_padding_mask)

        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        tgt_tokens = [bos_token_id]

        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(model.device)
            tgt_key_padding_mask = tgt_tensor == tokenizer.pad_token_id
            logits = model.decode(tgt_tensor, memory, src_key_padding_mask, tgt_key_padding_mask)
            next_token_id = logits.argmax(dim=-1)[0, -1].item()
            tgt_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return translation
