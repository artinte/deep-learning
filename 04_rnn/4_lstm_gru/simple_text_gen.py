import csv
import itertools
import nltk
import numpy

vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

nltk.download('punkt', quiet=True)

print('Read CSV file...')

with open('data/reddit-comments-2015-08.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, skipinitialspace=True)
    # Split full comments into sentences.
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
    # Append start and end tokens to each sentence.
    sentences = [sentence_start_token + ' ' + sentence + ' ' + sentence_end_token for sentence in sentences]

print('Parsed {} sentences.'.format(len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# Count the frequency of each word
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print('Found {} unique words.'.format(len(word_freq)))

# Get the most common words and build index_to_word and word_to_index mappings
vocab = word_freq.most_common(vocabulary_size -1)  # Reserve space for unknown and special tokens
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)  # Add unknown token
word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

print('Vocabulary size: {}'.format(len(index_to_word)))

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [word_to_index[w] for w in sent]
    
print('Example sentence: {}'.format(sentences[0]))
print('Example setence after tokenization: {}'.format(tokenized_sentences[0]))

# Create the training data
x_train = numpy.asarray([[word_to_index[w] for w in sentence[:-1]] for sentence in tokenized_sentences])
y_train = numpy.array([[word_to_index[w] for w in sentence[1:]] for sentence in tokenized_sentences])


