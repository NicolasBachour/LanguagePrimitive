
import sys
from itertools import islice

import parser

import numpy as np # DEBUG
import NeuralNetwork as nn
import LookupTable as lt

def main(argv):
    if (len(argv) < 3):
        print("Please specify the folder containing the dataset and the file containing the word2vec dictionnary :")
        print("main.py <dataset> <word2vec>")
        return

    ### Loading dataset ###
    print("Loading dataset...")
    data = parser.load_dataset(argv[1])

    ### Loading word vectors ###
    print("Loading word vectors...")
    word2vec = lt.LookupTable()
    word2vec.load(argv[2])

    ### Create Neural network, train it, and save it ###
    neural_network = nn.NeuralNetwork(word2vec.vector_dimention, 25)

    ### TESTS ###
    #sentence = [[0] * 300, [1] * 300, [2] * 300, [3] * 300, [4] * 300, [5] * 300, [6] * 300]
    #sentence = np.random.random([6, 300]).astype(np.float32)

    training_set_pos_length = len(data["neg"]) / 100
    training_set_neg_length = len(data["pos"]) / 100

    training_data = list(islice(data["neg"], training_set_neg_length)) + list(islice(data["pos"], training_set_pos_length))
    training_set = []
    for sentence in training_data:
        training_set.append(word2vec.convertSentence(sentence))

    neural_network.set_training_set(training_set,
                                    ([[0.0, 1.0]] * training_set_neg_length) + ([[1.0, 0.0]] * training_set_pos_length),
                                    training_set_neg_length + training_set_pos_length
                                    )
    #neural_network.train()

    sentence = ""
    while sentence != "EXIT":
        sentence = raw_input("Enter sentence : ")
        svector = word2vec.convertSentence(sentence)
        neural_network.run(svector)
    #############

    ### Map words in sentences to word vectors ###
#    i = 0
#    for sentence in data[neg]:
#        word2vec.convertSentence()
#        neural_network.run(sentence)
#        i += 1 # DEBUG
#        if (i == 4)
#            break

    #neural_network.train(vector_input_data, [1, 0]) # FUTURE
    #neural_network.save("./SAVES") # FUTURE
    return

if __name__ == "__main__":
    main(sys.argv)
