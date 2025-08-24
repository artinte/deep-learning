import torch


def translate(model, sentence, tokenizer, max_len=512):
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer(
            sentence, truncation=True, padding=False, return_tensors="pt"
        )
        src = src_tokens["input_ids"].to(model.device)
        src_key_padding_mask = (src_tokens["attention_mask"] == 0).to(model.device)
        memory = model.encode(src, src_key_padding_mask)

        bos_token_id = tokenizer.eos_token_id
        tgt_tokens = [bos_token_id]

        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(model.device)
            logits = model.decode(tgt_tensor, memory, src_key_padding_mask)
            next_token_id = logits.argmax(dim=-1)[0, -1].item()
            tgt_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return translation
