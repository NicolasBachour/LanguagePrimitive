
import numpy as np

class LookupTable:
    def __init__(self):
        self.table = dict()
        self.vector_dimension = 0
        return

    def load(self, filename):

        file = open(filename, "rb")
        # File properties
        dictionnary_size, self.vector_dimension = map(int, file.readline().split())

        vector_size = np.dtype('float32').itemsize * self.vector_dimension

        for i in xrange(0, dictionnary_size):
            word = []
            while True:
                ch = file.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            self.table[word] = np.fromstring(file.read(vector_size), dtype='float32')
#        self.table = word_vecs
        return

    def lookup(self, word):
        # What to do in case of error ?
        return self.table[word]
