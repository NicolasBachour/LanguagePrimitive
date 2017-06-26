
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

    integer_sentences = [[], []]
    max_sentence_length = max(len(line) for label in data for file in data[label] for line in data[label][file])
    print("Max sentence length is {0}".format(max_sentence_length))
    i = 0
    for label in data:
        for file in data[label]:
            for line in data[label][file]:
                integer_sentences[i].append(word2vec.convertSentence(line, max_sentence_length))
        i += 1

    # Create cv set
    print("Creating CV set...")
    positive_reviews = len(integer_sentences[0])
    negative_reviews = len(integer_sentences[0])
    cv_reviews = (positive_reviews + negative_reviews) / 10

    print("\tFound {0} positive reviews...".format(positive_reviews))
    print("\tFound {0} negative reviews...".format(negative_reviews))
    print("\tPicking {0} reviews from the set for cross-validation...".format(cv_reviews))

    training_set_input = []
    training_set_target = []
    validation_set_input = []
    validation_set_target = []
    random.seed()
    for i in xrange(cv_reviews):
        sample_class = random.randint(0, len(integer_sentences) - 1)
        sentence = random.randint(0, len(integer_sentences[sample_class]) - 1)

        #print("{0} : {1}".format(i, integer_sentences[sample_class][sentence]))

        training_set_input.append(integer_sentences[sample_class][sentence])
        training_set_target.append([1.0, 0.0] if sample_class == 0 else [0.0, 1.0])

        del integer_sentences[sample_class][sentence]

    for class_idx, sample_class in enumerate(integer_sentences):
        for sentence in xrange(len(sample_class)):
            validation_set_input.append(sample_class[sentence])
            validation_set_target.append([1.0, 0.0] if sample_class == 0 else [0.0, 1.0])

    ### Create Neural network, train it, and save it ###
    neural_network = nn.NeuralNetwork(number_of_classes = 2,
                                      vector_dimension = word2vec.getWordDimension(),
                                      vocabulary_size = word2vec.getVocabularySize(),
                                      sentence_length = max_sentence_length,
                                      dictionnary = word2vec.getLookupTable())

    ### TESTS ###
    print("Accuracy tests")
    neural_network.accuracy(training_set_input, training_set_target)
    neural_network.accuracy(validation_set_input, validation_set_target)

    neural_network.set_training_set(training_set_input, training_set_target, 50)

    print("Accuracy tests")
    neural_network.accuracy(training_set_input, training_set_target)
    neural_network.accuracy(validation_set_input, validation_set_target)

    #neural_network.save("./SAVES") # FUTURE
    return

if __name__ == "__main__":
    main(sys.argv)
