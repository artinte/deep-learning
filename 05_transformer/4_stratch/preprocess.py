import datasets
import torchtext
import nltk


def create_torchtext_dataset(raw_data, field_src, field_tgt):
    fields = [("src", field_src), ("tgt", field_tgt)]
    examples = [
        torchtext.data.Example.fromlist([src, tgt], fields=fields)
        for src, tgt in raw_data
    ]
    return torchtext.data.Dataset(examples, fields)


def preprocess(device, batch_size=32):
    nltk.download("punkt")
    dataset = datasets.load_dataset("bentrevett/multi30k")
    print(dataset)
    # {'en': 'Two young, White males are outside near many bushes.',
    # 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
    print(f"First train sample: {dataset["train"][0]}")

    train_data = [(example["de"], example["en"]) for example in dataset["train"]]
    valid_data = [(example["de"], example["en"]) for example in dataset["validation"]]
    test_data = [(example["de"], example["en"]) for example in dataset["test"]]

    src_field = torchtext.data.Field(
        tokenize=nltk.tokenize.word_tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        batch_first=True,
    )
    tgt_field = torchtext.data.Field(
        tokenize=nltk.tokenize.word_tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        batch_first=True,
    )

    train_dataset = create_torchtext_dataset(train_data, src_field, tgt_field)
    valid_dataset = create_torchtext_dataset(valid_data, src_field, tgt_field)
    test_dataset = create_torchtext_dataset(test_data, src_field, tgt_field)

    src_field.build_vocab(train_dataset, min_freq=2)
    tgt_field.build_vocab(train_dataset, min_freq=2)

    print("Source vocabulary size: " + str(len(src_field.vocab)))
    print("Target vocabulary size: " + str(len(tgt_field.vocab)))

    print(f"Source vocab examples: {list(src_field.vocab.stoi.keys())[:20]}")
    print(f"Target vocab examples: {list(tgt_field.vocab.stoi.keys())[:20]}")

    train_iterator = torchtext.data.BucketIterator(
        train_dataset,
        batch_size=batch_size,
        device=device,
        sort_within_batch=True,
        sort_key=lambda x: len(x.src),
    )

    valid_iterator, test_iterator = torchtext.data.BucketIterator.splits(
        (valid_dataset, test_dataset),
        batch_size=batch_size,
        device=device,
        sort_within_batch=False
    )

    for batch in train_iterator:
        src = batch.src
        tgt = batch.tgt

        print(src.shape)
        print(tgt.shape)
        break

    return src_field, tgt_field, train_iterator, valid_iterator, test_iterator
