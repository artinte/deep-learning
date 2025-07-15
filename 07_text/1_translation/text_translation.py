import datasets

dataset = datasets.load_dataset('ted_hrlr', 'pt_to_en', trust_remote_code=True)
train_examples = dataset["train"]
val_examples = dataset["validation"]

print(f"Train examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")

print(train_examples[0])