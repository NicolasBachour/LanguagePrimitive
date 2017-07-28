
import sys
import os
from common import parser
from common import NeuralNetwork as nn
from common import LookupTable as lt

MAXIMISER_INSTANCES = 1 # Should be 9

def main(argv):
    if (len(argv) < 4):
        print("Please specify the dataset, the file containing the word2vec dictionnary, and the folder containing the trained weights of the network  :")
        print("main.py <dataset> <word2vec> <save>")
        return

    sentence_set = {
        "char" : [],
        "input" : [],
        "target" : []
    }

    print("Loading word vectors...")
    with lt.LookupTable() as word2vec:
        word2vec.load(argv[2])

        ### Loading dataset ###
        print("Loading dataset...")
        data = parser.load_dataset(argv[1])

        max_sentence_length = max(len(line) for label in data for file in data[label] for line in data[label][file])
        print("Max sentence length is {0}. Sentences will be padded to this size to make the use of batches possible.".format(max_sentence_length))

        print("Storing sentences, inputs, and targets...")
        for i, label in enumerate(data):
            for file in data[label]:
                for line in data[label][file]:
                    sentence_set["char"].append(line)
                    sentence_set["input"].append(word2vec.convertSentence(line, max_sentence_length))
                    sentence_set["target"].append([1.0, 0.0] if i == 0 else [0.0, 1.0])

        print("Creating static network...")
        neural_network = nn.NeuralNetwork(number_of_classes = 2,
                                          vector_dimension = word2vec.getWordDimension(),
                                          vocabulary_size = word2vec.getVocabularySize(),
                                          sentence_length = max_sentence_length,
                                          dictionnary = word2vec.getLookupTable(),
                                          is_trainable = False)

        print("Restoring weights...")
        neural_network.restore(os.path.join(argv[3], "Save"))

        #-------------------------

        #print("Searching for neurons with highly variable activation...")
        #neuron_x, neuron_y = neural_network.search_variable_neuron()

        #for x in xrange(5, 6):
        #    for y in xrange(1):
        #        print("Maximising neuron...")
        #        maximiser = neural_network.create_kernel_maximiser(MAXIMISER_INSTANCES, 5, 0)
        #        maximiser.run()
        #        sentence = maximiser.find_maximising_sentence(word2vec)
        #        print(sentence)

        while True:
            x = int(input("Kernel size : "))
            y = int(input("Index : "))
            print("Maximising neuron...")
            maximiser = neural_network.create_kernel_maximiser(MAXIMISER_INSTANCES, x, y)
            maximiser.run()
            sentence = maximiser.find_maximising_sentence(word2vec)
            print(sentence)
    return


if __name__ == "__main__":
    main(sys.argv)