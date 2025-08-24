import torch


def translate(model, sentence, tokenizer, max_len=512):
    model.eval()
    with torch.no_grad():
        src_tensor = tokenizer(
            sentence, truncation=True, padding=False, return_tensors="pt"
        ).to(model.device)

        memory = model.encode(src_tensor)

        bos_token_id = tokenizer.pad_token_id
        tgt_tokens = [bos_token_id]

        for i in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(model.device)
            logits = model.decode(tgt_tensor, memory)
            next_token_id = logits.argmax(dim=-1)[0, -1].item()
            tgt_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

    translation = tokenizer.decode(tgt_tokens, skip_special_tokens=True)
    return translation
