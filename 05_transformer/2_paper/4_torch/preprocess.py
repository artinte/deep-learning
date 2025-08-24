import datasets
import transformers
import torch


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        en_text = self.dataset[index]["en"]
        de_text = self.dataset[index]["de"]
        return en_text, de_text


def preprocess(tokenizer):
    data_train, data_valid, data_test = datasets.load_dataset(
        "bentrevett/multi30k", split=["train", "validation", "test"]
    )
    print(f"Training dataset length: {len(data_train)}")
    print(f"Validation dataset length: {len(data_valid)}")
    print(f"Test dataset length: {len(data_test)}")
    print(f"First train sample: {data_train[0]}")
    
    first_sample_eng_token = tokenizer(
        data_train[0]["en"],
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    print(f"First sample token IDs (Source): {first_sample_eng_token['input_ids']}")
    first_sample_de_token = tokenizer(
        data_train[0]["de"],
        max_length=tokenizer.model_max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    print(f"First sample token IDs (Destination): {first_sample_de_token['input_ids']}")


    def collate_fn(batch):
        en_sentences = [item[0] for item in batch]
        de_sentences = [item[1] for item in batch]

        src_tokens = tokenizer(
            en_sentences, truncation=True, padding=True, return_tensors="pt"
        )
        tgt_tokens = tokenizer(
            de_sentences, truncation=True, padding=True, return_tensors="pt"
        )

        # The tokenizer now returns a dictionary with 'input_ids' and 'attention_mask'
        # We only need the input IDs for this model.
        return src_tokens["input_ids"], tgt_tokens["input_ids"]

    BATCH_SIZE = 16

    train_dataset = TranslationDataset(data_train)
    valid_dataset = TranslationDataset(data_valid)
    test_dataset = TranslationDataset(data_test)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    test_src_sample, test_tgt_sample = next(iter(test_dataloader))
    print(f"Shape of test src sample: {test_src_sample.shape}")
    print(f"First batch test src token: {test_src_sample}")
    print(f"Shape of test tgt sample: {test_tgt_sample.shape}")
    print(f"First batch test tgt token: {test_tgt_sample}")

    return train_dataloader, valid_dataloader, test_dataloader
