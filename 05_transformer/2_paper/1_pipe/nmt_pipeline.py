import datasets
import transformers

dataset = datasets.load_dataset("bentrevett/multi30k", split="test")

translator = transformers.pipeline('translation_en_to_de',
                                   model="Helsinki-NLP/opus-mt-en-de")

for i in range(32):
    en_sentence = datasets[i]['en']
    de_reference = datasets[i]['de']
    de_pred = translator(en_sentence)[0]["translation_text"]

    print("-" * 50)
    print(f"Source: {en_sentence}")
    print(f"Prediction: {de_pred}")
    print(f"Reference: {de_reference}")
