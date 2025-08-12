import datasets
import tiktoken


enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    dataset = datasets.load_dataset("openwebtext", num_proc=8)

    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
