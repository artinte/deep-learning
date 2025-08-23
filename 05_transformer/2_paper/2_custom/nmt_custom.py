import datasets
import torch
import transformers

dataset = datasets.load_dataset("bentrevett/multi30k", split="test")

model_name = "Helsinki-NLP/opus-mt-en-de"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

model.eval()

for i in range(32):
    en_sentence = dataset[i]["en"]
    de_reference = dataset[i]["de"]

    inputs = tokenizer(en_sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # Customize the max length of the output
            max_length=256,
            # Use beam search for better quality
            num_beams=4,
            early_stopping=True,
        )

    de_pred = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {de_pred}")
    print(f"Reference: {de_reference}")
