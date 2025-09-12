import unittest
from data_wiki_text import Dictionary, Corpus


class TestDictionary(unittest.TestCase):
    """
    Test suite for the Dictionary class.
    """

    def setUp(self):
        """
        Set up a fresh Dictionary instance before each test case.
        """
        self.dictionary = Dictionary()

    def test_initialization(self):
        """
        Test that the dictionary is initialized with empty word mappings.
        """
        self.assertEqual(self.dictionary.word2idx, {})
        self.assertEqual(self.dictionary.idx2word, [])
        self.assertEqual(len(self.dictionary), 0)

    def test_add_new_word(self):
        """
        Test adding a new word and verify its index and mappings.
        """
        word1 = "hello"
        index1 = self.dictionary.add_word(word1)

        self.assertEqual(index1, 0)
        self.assertEqual(len(self.dictionary), 1)
        self.assertIn(word1, self.dictionary.word2idx)
        self.assertEqual(self.dictionary.word2idx[word1], 0)
        self.assertEqual(self.dictionary.idx2word[0], word1)

    def test_add_multiple_words(self):
        """
        Test adding multiple unique words sequentially.
        """
        words = ["the", "quick", "brown", "fox"]
        for i, word in enumerate(words):
            index = self.dictionary.add_word(word)
            self.assertEqual(index, i)

        self.assertEqual(len(self.dictionary), 4)
        self.assertEqual(self.dictionary.idx2word, words)
        for i, word in enumerate(words):
            self.assertEqual(self.dictionary.word2idx[word], i)

    def test_add_existing_word(self):
        """
        Test that adding an existing word does not change the dictionary's size or mappings.
        """
        self.dictionary.add_word("apple")
        self.dictionary.add_word("banana")

        initial_len = len(self.dictionary)

        # Add a word that already exists
        existing_index = self.dictionary.add_word("apple")

        # Verify that the returned index is correct and the length has not changed
        self.assertEqual(existing_index, 0)
        self.assertEqual(len(self.dictionary), initial_len)
        self.assertEqual(self.dictionary.idx2word, ["apple", "banana"])
        self.assertEqual(self.dictionary.word2idx["apple"], 0)


wiki_corpus = Corpus("data/wiki_text")
print(len(wiki_corpus.dictionary))

if __name__ == "__main__":
    unittest.main()
