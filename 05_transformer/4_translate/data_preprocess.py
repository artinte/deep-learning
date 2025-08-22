import datasets
import torchtext
import nltk


def preprocess(device):
    nltk.download("punkt")
    dataset = datasets.load_dataset("bentrevett/multi30k")
    print(dataset)
    # {'en': 'Two young, White males are outside near many bushes.',
    # 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
    print(dataset["train"][0])

    train_data = [(example["de"], example["en"]) for example in dataset["train"]]
    valid_data = [(example["de"], example["en"]) for example in dataset["validation"]]
    test_data = [(example["de"], example["en"]) for example in dataset["test"]]

    src_field = torchtext.data.Field(
        tokenize=nltk.tokenize.word_tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        lower=True,
        batch_first=True,
    )
    trg_field = torchtext.data.Field(
        tokenize=nltk.tokenize.word_tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        lower=True,
        batch_first=True,
    )

    train_examples = [
        torchtext.data.Example.fromlist(
            [src, trg], fields=[("src", src_field), ("trg", trg_field)]
        )
        for src, trg in train_data
    ]
    valid_examples = [
        torchtext.data.Example.fromlist(
            [src, trg], fields=[("src", src_field), ("trg", trg_field)]
        )
        for src, trg in valid_data
    ]
    test_examples = [
        torchtext.data.Example.fromlist(
            [src, trg], fields=[("src", src_field), ("trg", trg_field)]
        )
        for src, trg in test_data
    ]

    train_dataset = torchtext.data.Dataset(
        examples=train_examples, fields=[("src", src_field), ("trg", trg_field)]
    )
    valid_dataset = torchtext.data.Dataset(
        examples=valid_examples, fields=[("src", src_field), ("trg", trg_field)]
    )
    test_dataset = torchtext.data.Dataset(
        examples=test_examples, fields=[("src", src_field), ("trg", trg_field)]
    )

    src_field.build_vocab(train_dataset, min_freq=2)
    trg_field.build_vocab(train_dataset, min_freq=2)

    print("Source vocabulary size: " + str(len(src_field.vocab)))
    print("Target vocabulary size: " + str(len(trg_field.vocab)))

    print([word for word, _ in list(src_field.vocab.stoi.items())[:10]])
    print([word for word, _ in list(trg_field.vocab.stoi.items())[:10]])
    
    train_iterator, valid_iterator, test_iterator = (
        torchtext.data.BucketIterator.splits(
            (train_dataset, valid_dataset, test_dataset),
            batch_size=32,
            device=device,
            sort_within_batch=True,
            sort_key=lambda x: len(x.src),
        )
    )

    for batch in train_iterator:
        src = batch.src
        trg = batch.trg

        print(src.shape)
        print(trg.shape)
        break

    return src_field, trg_field, train_iterator, valid_iterator, test_iterator
