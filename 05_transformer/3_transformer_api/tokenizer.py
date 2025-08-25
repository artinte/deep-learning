import transformers


def get_tokenizer():
    tokenizer = transformers.AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    print(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    print(f"Unknown token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
    print(f"Max truncation length: {tokenizer.model_max_length}")

    return tokenizer
