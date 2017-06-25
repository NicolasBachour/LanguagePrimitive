
import sys
import random

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
    neural_network = nn.NeuralNetwork(word2vec.vector_dimension, 25)

    ### TESTS ###
    #sentence = [[0] * 300, [1] * 300, [2] * 300, [3] * 300, [4] * 300, [5] * 300, [6] * 300]
    #sentence = np.random.random([6, 300]).astype(np.float32)

    positive_reviews = sum(len(file) for file in data["neg"])
    negative_reviews = sum(len(file) for file in data["pos"])
    cv_reviews = (positive_reviews + negative_reviews) / 10

    print("Reading {0} positive reviews...".format(positive_reviews))
    print("Reading {0} negative reviews...".format(negative_reviews))
    print("Picking {0} reviews from the set for cross-validation...".format(cv_reviews))

#    training_data = []
    training_set_input = []
    training_set_target = []
    random.seed()
    for i in xrange(cv_reviews):
        sample_class = random.choice(data.keys())
        sample = random.choice(data[sample_class].keys())
        sentence = random.randint(0, len(data[sample_class][sample]) - 1)

        #print("{0} : {1}".format(i, data[sample_class][sample][sentence]))
        #training_data.append(data[sample_class][sample][sentence])
        training_set_input.append(word2vec.convertSentence(data[sample_class][sample][sentence]))
        training_set_target.append([1.0, 0.0] if sample_class == "pos" else [0.0, 1.0])
        #DEBUG ?
        del data[sample_class][sample][sentence]
        if len(data[sample_class][sample]) == 0:
            del data[sample_class][sample]


#    training_data = list(islice(data["neg"], training_set_neg_length)) + list(islice(data["pos"], training_set_pos_length))

#    for sample in training_data:
#        print(sample)
#        for sentence in sample:
#            #print(sentence)
#            training_set.append(word2vec.convertSentence(sentence))

    neural_network.set_training_set(training_set_input, training_set_target, 1)

    validation_set_input = []
    validation_set_target = []
    for x in data["pos"]:
        for sentence in x:
            validation_set_input.append(word2vec.convertSentence(sentence))
            validation_set_target.append([1.0, 0.0])
    for x in data["neg"]:
        for sentence in x:
            validation_set_input.append(word2vec.convertSentence(sentence))
            validation_set_target.append([0.0, 1.0])

    neural_network.accuracy(training_set_input, training_set_target)
    neural_network.accuracy(validation_set_input, validation_set_target)

#    sentence = ""
#    while sentence != "EXIT":
#        sentence = raw_input("Enter sentence : ")
#        svector = word2vec.convertSentence(sentence)
#        neural_network.run(svector)
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
