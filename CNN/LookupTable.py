
import numpy as np
from collections import defaultdict

class LookupTable:
    def __init__(self):
        self.table = defaultdict(lambda: np.random.random([self.vector_dimension]).astype(np.float32) * 2 - 1)
        self.vector_dimension = 0
        return

    def load(self, filename):

        file = open(filename, "rb")
        # File properties
        dictionnary_size, self.vector_dimension = map(int, file.readline().split())

        vector_size = np.dtype('float32').itemsize * self.vector_dimension

        for i in xrange(0, dictionnary_size):
            word = ""
            while True:
                c = file.read(1)
                if c == ' ':
                    break
                if c != '\n':
                    word += c
            self.table[word] = np.fromstring(file.read(vector_size), dtype='float32')
        return

    def lookup(self, word):
        return self.table[word]

    def convertSentence(self, sentence):
        sentence_vectors = []
        words = sentence.split()
        for w in words:
            sentence_vectors.append(self.table[w])
        return sentence_vectors