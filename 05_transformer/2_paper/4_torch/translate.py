import torch

"""
This file contains two main functions for machine translation inference:
one using a simple but limited greedy search, and a second, more advanced
one using beam search with length normalization for higher quality output.
"""

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


def get_normalized_score(score, tokens, alpha=0.8, smoothing_constant=5.0):
    """
    Calculates a length-normalized score for a translation hypothesis.

    This function applies a length penalty to prevent the beam search algorithm
    from favoring short, incomplete sequences. It's based on the common Google
    Neural Machine Translation (GNMT) length normalization formula.

    Args:
        score (float): The raw accumulated log probability score of the sequence.
        tokens (list): The list of token IDs for the sequence.
        alpha (float): The length penalty exponent. A higher value penalizes
                       longer sentences more. Common values are 0.6 to 0.8.
        smoothing_constant (float): A constant added to the length to avoid
                                    division by zero and to smooth the penalty.

    Returns:
        float: The length-normalized score.
    """
    length = len(tokens)
    denominator = ((length + smoothing_constant) ** alpha) / (
        (1.0 + smoothing_constant) ** alpha
    )
    return score / denominator


def translate_beam_search(model, sentence, tokenizer, max_len=512, num_beams=5):
    """
    Performs machine translation using a beam search algorithm.

    Beam search explores multiple promising translation hypotheses simultaneously.
    At each step, it keeps track of the 'num_beams' most likely partial
    sequences, eventually selecting the best complete translation based on a
    length-normalized score. This method significantly improves translation
    quality over greedy search.

    Args:
        model (nn.Module): The trained Transformer translation model.
        sentence (str): The source sentence to translate.
        tokenizer: The tokenizer used for encoding/decoding text.
        max_len (int): The maximum length of the output sequence.
        num_beams (int): The number of candidate sequences (beams) to maintain
                         at each decoding step. A higher value leads to better
                         quality but requires more computation.

    Returns:
        str: The translated sentence.
    """
    model.eval()
    with torch.no_grad():
        src_tokens = tokenizer(
            sentence, truncation=True, padding=False, return_tensors="pt"
        )
        src = src_tokens["input_ids"].to(model.device)
        src_key_padding_mask = (src_tokens["attention_mask"] == 0).to(model.device)
        memory = model.encode(src, src_key_padding_mask)

        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id

        beams = [(0.0, [bos_token_id])]
        final_candidates = []

        for _ in range(max_len):
            all_candidates = []

            for score, tokens in beams:
                if tokens[-1] == tokenizer.eos_token_id:
                    final_candidates.append((score, tokens))
                    continue

                tgt_tensor = torch.LongTensor(tokens).unsqueeze(0).to(model.device)

                logits = model.decode(tgt_tensor, memory, src_key_padding_mask)[
                    :, -1, :
                ]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)
                topk_log_probs, topk_indices = torch.topk(log_probs, num_beams)

                for i in range(num_beams):
                    new_token_id = topk_indices[i].item()
                    new_score = score + topk_log_probs[i].item()
                    new_tokens = tokens + [new_token_id]

                    all_candidates.append((new_score, new_tokens))

            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:num_beams]
            
            if all(tokens[-1] == tokenizer.eos_token_id for _, tokens in beams):
                break

            if not beams:
                break

        final_candidates.extend(beams)

        if not final_candidates:
            final_candidates = all_candidates

        best_candidate = max(
            final_candidates, key=lambda x: get_normalized_score(x[0], x[1])
        )
        best_translation_tokens = best_candidate[1]

    translation = tokenizer.decode(best_translation_tokens, skip_special_tokens=True)
    return translation
