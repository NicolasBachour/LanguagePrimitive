
import sys
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

    #for line in content:
    #    print("\t" + line)
    #content = [x.strip() for x in content]
    #print("\t" + filename + " in " + folder)

    ### Loading word vectors ###
    print("Loading word vectors...")
    word2vec = lt.LookupTable()
    word2vec.load(argv[2])

    ### Map words in sentences to word vectors ###

    ### Create Neural network, train it, and save it ###
    neural_network = nn.NeuralNetwork()

    ### TESTS ###
    #sentence = [[0] * 300, [1] * 300, [2] * 300, [3] * 300, [4] * 300, [5] * 300, [6] * 300]
    sentence = np.random.random([6, 300]).astype(np.float32)
    #############

    neural_network.run(sentence)
    #neural_network.train(vector_input_data) # FUTURE
    #neural_network.save("./SAVES") # FUTURE
    return

if __name__ == "__main__":
    main(sys.argv)
