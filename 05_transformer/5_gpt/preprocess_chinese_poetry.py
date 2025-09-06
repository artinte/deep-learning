import pandas
from collections import Counter

special_tokens = {
    "<pad>": 0,
    "<unk>": 1,
    "<sos>": 2,
    "<eos>": 3,
}


def proprocess(file_path):
    print("Preprocessing Chinese Poetry Dataset...")
    df = pandas.read_csv(file_path, header=None)
    print("Original length:", len(df))
    print("First poem:", df.iloc[0, 0])

    df.rename(columns={0: "poem"}, inplace=True)
    df.dropna(subset=["poem"], inplace=True)
    df["poem"] = df["poem"].astype(str).str.strip()
    df = df[df["poem"].str.len() >= 5]

    allowed_chars_pattern = r"^[，。？！；：\u4e00-\u9fa5]+$"
    df = df[df["poem"].astype(str).str.match(allowed_chars_pattern, na=False)]
    print("Cleaned length:", len(df))

    tokenized_poems = [list(poem) for poem in df["poem"] if poem]

    word_counts = Counter(token for poem in tokenized_poems for token in poem)
    vocab = {
        token: i + len(special_tokens)
        for i, (token, _) in enumerate(word_counts.items())
    }
    vocab.update({token: i for token, i in special_tokens.items()})

    idx_to_word = {idx: word for word, idx in vocab.items()}

    def text_to_ids(poem_tokens):
        ids = (
            [vocab["<sos>"]]
            + [vocab.get(token, vocab["<unk>"]) for token in poem_tokens]
            + [vocab["<eos>"]]
        )
        return ids

    encoded_poems = [text_to_ids(tokens) for tokens in tokenized_poems]
    return encoded_poems, vocab, idx_to_word


proprocess(file_path="data/chinese_poetry.csv")
