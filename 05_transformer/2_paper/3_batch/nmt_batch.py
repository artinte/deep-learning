import datasets
import torch
import transformers

dataset = datasets.load_dataset("bentrevett/multi30k", split="test")

model_name = "Helsinki-NLP/opus-mt-en-de"
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

model.eval()

en_sentences = [dataset[i]["en"] for i in range(32)]
de_references = [dataset[i]["de"] for i in range(32)]

inputs = tokenizer(en_sentences, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    translated_tokens = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=256,
        num_beams=4,
        early_stopping=True,
    )

de_predictions = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

for i in range(32):
    print("-" * 50)
    print(f"Source: {en_sentences[i]}")
    print(f"Prediction: {de_predictions[i]}")
    print(f"Reference: {de_references[i]}")
