import datasets
import torchtext
import nltk


def preprocess():
    nltk.download("punkt")
    dataset = datasets.load_dataset("bentrevett/multi30k")
    print(dataset)
    # {'en': 'Two young, White males are outside near many bushes.',
    # 'de': 'Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche.'}
    print(dataset["train"][0])

    train_data = [(example["de"], example["en"]) for example in dataset["train"]]
    valid_data = [(example["de"], example["en"]) for example in dataset["validation"]]
    test_data = [(example["de"], example["en"]) for example in dataset["test"]]

    SRC = torchtext.data.Field(
        tokenize=nltk.tokenize.word_tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        lower=True,
        batch_first=True,
    )
    TRG = torchtext.data.Field(
        tokenize=nltk.tokenize.word_tokenize,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        lower=True,
        batch_first=True,
    )

    train_examples = [
        torchtext.data.Example.fromlist([src, trg], fields=[("src", SRC), ("trg", TRG)])
        for src, trg in train_data
    ]
    valid_examples = [
        torchtext.data.Example.fromlist([src, trg], fields=[("src", SRC), ("trg", TRG)])
        for src, trg in valid_data
    ]
    test_examples = [
        torchtext.data.Example.fromlist([src, trg], fields=[("src", SRC), ("trg", TRG)])
        for src, trg in test_data
    ]

    train_dataset = torchtext.data.Dataset(
        examples=train_examples, fields=[("src", SRC), ("trg", TRG)]
    )
    valid_dataset = torchtext.data.Dataset(
        examples=valid_examples, fields=[("src", SRC), ("trg", TRG)]
    )
    test_dataset = torchtext.data.Dataset(
        examples=test_examples, fields=[("src", SRC), ("trg", TRG)]
    )

    SRC.build_vocab(train_dataset, min_freq=2)
    TRG.build_vocab(train_dataset, min_freq=2)

    print("Source vocabulary size: " + str(len(SRC.vocab)))
    print("Target vocabulary size: " + str(len(TRG.vocab)))

    print([word for word, _ in list(SRC.vocab.stoi.items())[:10]])
    print([word for word, _ in list(TRG.vocab.stoi.items())[:10]])

    return SRC, TRG, train_dataset, valid_dataset, test_dataset
