
# A simple class which serves as an interface to the word2vec dictionnary, also handling integer association and new vocabulary integration #
# - Created by Nicolas Bachour, 2017
# ------------------------ #

import numpy as np

class LookupTable:
    def __init__(self):
        self.word_to_int = dict()
        self.int_to_word = dict()
        self.lookup_table = []
        self.vector_dimension = 0
        return

    def __enter__(self):
        return self
    def __exit__(self, *err):
        del self.lookup_table
        del self.int_to_word
        del self.word_to_int
        return

    def load(self, filename):

        file = open(filename, "rb")
        # File properties
        dictionnary_size, self.vector_dimension = map(int, file.readline().split())

        vector_size = np.dtype('float32').itemsize * self.vector_dimension

        self.lookup_table.append(np.zeros(self.vector_dimension, np.float32))
        for i in xrange(0, dictionnary_size):
            word = ""
            while True:
                c = file.read(1)
                if c == ' ':
                    break
                if c != '\n':
                    word += c
            self.word_to_int[word] = i
            self.int_to_word[i] = word
            self.lookup_table.append(np.fromstring(file.read(vector_size), dtype='float32'))
        return

    def lookup(self, word):
        return self.lookup_table[self.word_to_int[word.lower()]]

    def reverse_lookup(self, word_as_int):
        return self.int_to_word[word_as_int]

    def lookup_int(self, word_as_int):
        return self.lookup_table[word_as_int]

    def convertSentence(self, sentence, padded_length):
        sentence_integers = []
        for base_word in sentence:
            word = base_word.lower()
            if not word in self.word_to_int:
                index = len(self.lookup_table)
                self.word_to_int[word] = index
                self.int_to_word[index] = word
                self.lookup_table.append(np.random.uniform(-0.5, 0.5, self.vector_dimension))
            sentence_integers.append(self.word_to_int[word])
        while len(sentence_integers) < padded_length:
            sentence_integers.append(0)
        return sentence_integers

    def getWordDimension(self):
        return self.vector_dimension

    def getLookupTable(self):
        return self.lookup_table

    def getVocabularySize(self):
        return len(self.lookup_table)