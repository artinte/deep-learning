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


def preprocess(device, batch_size=32, dataset=None):
    nltk.download("punkt")
    if dataset == None:
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

    print(f"Source <eos> index: {src_field.vocab.stoi[src_field.eos_token]}")
    print(f"Source <pad> index: {src_field.vocab.stoi[src_field.pad_token]}")
    print(f"Source <unk> index: {src_field.vocab.stoi[src_field.unk_token]}")
    print(f"Target <sos> index: {tgt_field.vocab.stoi[tgt_field.init_token]}")
    print(f"Target <eos> index: {tgt_field.vocab.stoi[tgt_field.eos_token]}")
    print(f"Target <pad> index: {tgt_field.vocab.stoi[tgt_field.pad_token]}")
    print(f"Target <unk> index: {tgt_field.vocab.stoi[tgt_field.unk_token]}")

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

    valid_iterator = torchtext.data.BucketIterator(
        valid_dataset,
        batch_size=batch_size,
        device=device,
        sort_within_batch=False,
    )

    test_iterator = torchtext.data.BucketIterator(
        test_dataset,
        batch_size=batch_size,
        device=device,
        sort_within_batch=False,
    )

    for batch in train_iterator:
        src = batch.src
        tgt = batch.tgt

        print(f"First src train sample shape: {src.shape}")
        print(f"First tgt train sample shape: {tgt.shape}")
        break

    for batch in test_iterator:
        test_src_sample = batch.src
        test_tgt_sample = batch.tgt
        test_src_mask = test_src_sample == src_field.vocab.stoi[src_field.pad_token]
        test_tgt_mask = test_tgt_sample == tgt_field.vocab.stoi[tgt_field.pad_token]
        print(f"Shape of test src sample: {test_src_sample.shape}")
        print(f"First batch test src token: {test_src_sample}")
        print(f"Shape of test tgt sample: {test_tgt_sample.shape}")
        print(f"First batch test tgt token: {test_tgt_sample}")
        print(f"Shape of test src mask: {test_src_mask.shape}")
        print(f"First batch test src mask: {test_src_mask}")
        print(f"Shape of test tgt mask: {test_tgt_mask.shape}")
        print(f"First batch test tgt mask: {test_tgt_mask}")
        break

    return (
        src_field,
        tgt_field,
        train_iterator,
        valid_iterator,
        test_iterator,
        test_data,
    )
