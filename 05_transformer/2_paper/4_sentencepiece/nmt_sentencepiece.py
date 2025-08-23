import sentencepiece
import transformers
import torch
import huggingface_hub
import datasets
from typing import List, Dict


def encode_with_sp(
    texts: list, tokenizer: sentencepiece.SentencePieceProcessor, max_length=256
):
    input_ids = tokenizer.EncodeAsIds(texts)
    padded_input_ids = []
    attention_masks = []

    for ids in input_ids:
        # truncate if the sequence is too long
        ids = ids[:max_length]
        pad_len = max_length - len(ids)
        # uses 0 for paddding
        attention_mask = [1] * len(ids) + [0] * pad_len
        padded_ids = ids + [0] * pad_len

        padded_input_ids.append(padded_ids)
        attention_masks.append(attention_mask)

    return {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(attention_mask),
    }


def decode_with_sp(
    token_ids: List[List[int]], tokenizer: sentencepiece.SentencePieceProcessor
) -> List[str]:
    return tokenizer.Decode(token_ids)


dataset = datasets.load_dataset("bentrevett/multi30k", split="test")

# Download the SentencePiece model file
model_file_path = huggingface_hub.hf_hub_download(
    repo_id="Helsinki-NLP/opus-mt-en-de", filename="source.spm"
)
sp_tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file_path)

model_name = "Helsinki-NLP/opus-mt-en-de"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

en_sentences = [dataset[i]["en"] for i in range(32)]
de_references = [dataset[i]["de"] for i in range(32)]

inputs = encode_with_sp(en_sentences, sp_tokenizer)

with torch.no_grad():
    translated_tokens = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        num_beams=4,
        early_stopping=True,
    )

decoded_preds = decode_with_sp(translated_tokens.tolist(), sp_tokenizer)

for i in range(32):
    print("-" * 50)
    print(f"Source: {en_sentences[i]}")
    print(f"Prediction: {decoded_preds[i]}")
    print(f"Reference: {de_references[i]}")
