import datasets
import transformers

dataset = datasets.load_dataset("bentrevett/multi30k", split="test")

translator = transformers.pipeline(
    task="translation_en_to_de", model="Helsinki-NLP/opus-mt-en-de", framework="pt"
)

for i in range(32):
    en_sentence = dataset[i]["en"]
    de_reference = dataset[i]["de"]
    de_pred = translator(en_sentence)[0]["translation_text"]

    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {de_pred}")
    print(f"Reference: {de_reference}")
