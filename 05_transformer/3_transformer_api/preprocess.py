import datasets
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


def preprocess(tokenizer, device, batch_size=16, dataset=None):
    if dataset:
        data_train, data_valid, data_test = (
            dataset["train"],
            dataset["validation"],
            dataset["test"],
        )
    else:
        data_train, data_valid, data_test = datasets.load_dataset(
            "bentrevett/multi30k", split=["train", "validation", "test"]
        )
        wmt_dataset = datasets.load_dataset("wmt14", "de-en", split="train[:3%]")
        wmt_dataset = wmt_dataset.map(lambda x: {"de": x["translation"]["de"], "en": x["translation"]["en"]})
        wmt_dataset = wmt_dataset.remove_columns("translation")
        data_train = datasets.concatenate_datasets([data_train, wmt_dataset])
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
        ).to(device)
        tgt_tokens = tokenizer(
            de_sentences, truncation=True, padding=True, return_tensors="pt"
        ).to(device)
        bos_token_id = tokenizer.bos_token_id or tokenizer.eos_token_id
        bos = torch.full(
            (tgt_tokens["input_ids"].size(0), 1), bos_token_id, dtype=torch.long
        ).to(device)
        bos_mask = torch.ones(
            (tgt_tokens["attention_mask"].size(0), 1), dtype=torch.long
        ).to(device)

        # The tokenizer now returns a dictionary with 'input_ids' and 'attention_mask'
        # We only need the input IDs for this model.
        return (
            src_tokens["input_ids"],
            torch.cat([bos, tgt_tokens["input_ids"]], dim=1),
            src_tokens["attention_mask"],
            torch.cat([bos_mask, tgt_tokens["attention_mask"]], dim=1),
        )

    train_dataset = TranslationDataset(data_train)
    valid_dataset = TranslationDataset(data_valid)
    test_dataset = TranslationDataset(data_test)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_src_sample, test_tgt_sample, test_src_mask, test_tgt_mask = next(
        iter(test_dataloader)
    )
    print(f"Shape of test src sample: {test_src_sample.shape}")
    print(f"First batch test src token: {test_src_sample}")
    print(f"Shape of test tgt sample: {test_tgt_sample.shape}")
    print(f"First batch test tgt token: {test_tgt_sample}")
    print(f"Shape of test src mask: {test_src_mask.shape}")
    print(f"First batch test src mask: {test_src_mask}")
    print(f"Shape of test tgt mask: {test_tgt_mask.shape}")
    print(f"First batch test tgt mask: {test_tgt_mask}")

    return train_dataloader, valid_dataloader, test_dataloader, data_test
