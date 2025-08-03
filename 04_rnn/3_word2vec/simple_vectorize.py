
sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
print(f'Token length of sentence ({sentence}):', len(tokens))

vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1
vocab_size = len(vocab)
print('Vocab', vocab)

inverse_vocab = {index: token for token, index in vocab.items()}
print('Inverse vocab', inverse_vocab)

example_sequence = [vocab[word] for word in tokens]
print('Tokenized sentence:', example_sequence)


def skipgrams_torch(sequence, window_size=2):
    skip_grams = []
    for i, target_word in enumerate(sequence):
        # Find context words within the window
        context_start = max(0, i - window_size)
        context_end = min(len(sequence), i + window_size + 1)
        
        # Iterate through the context words and create pairs.
        for j in range(context_start, context_end):
          if i != j:  # Avoid pairing the target word with itself
            context_word = sequence[j]
            skip_grams.append((target_word, context_word))
        
    return skip_grams
  

window_size = 2
positive_skip_grams = skipgrams_torch(example_sequence, window_size)
print('Number of positive skip-grams:', len(positive_skip_grams))
for target, context in positive_skip_grams:
    print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")


# Get target and context words for one positive skip-gram.
target_word, context_word = positive_skip_grams[0]

# Set the number of negative samples per positive context.
num_ns = 4
