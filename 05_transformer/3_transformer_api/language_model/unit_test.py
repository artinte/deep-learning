from data_wiki_text import Dictionary, Corpus

wiki_corpus = Corpus("data/wiki_text")
print(len(wiki_corpus.dictionary))
