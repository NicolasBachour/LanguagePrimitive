
import numpy as np

class LookupTable:
    def __init__(self):
        self.word_to_int = dict()
        self.lookup_table = []
        self.vector_dimension = 0
        return

    def __enter__(self):
        return self
    def __exit__(self, *err):
        del self.lookup_table
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
            self.lookup_table.append(np.fromstring(file.read(vector_size), dtype='float32'))
        return

    def lookup(self, word):
        return self.lookup_table[word]

    def convertSentence(self, sentence, padded_length):
        sentence_integers = []
        for word in sentence:
            if not word in self.word_to_int:
                self.word_to_int[word] = len(self.lookup_table)
                self.lookup_table.append(np.random.random(self.vector_dimension))
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