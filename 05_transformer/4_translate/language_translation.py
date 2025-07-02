import os
import spacy

def load_tokenizers():
    try:
        spacy_de = spacy.load('de_core_news_sm')
    except IOError:
        os.system('python -m spacy download de_core_news_sm')
        spacy_de = spacy.load('de_core_news_sm')
        
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en

def toeknize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])
        

# global variables used later in the script
        
