
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

    random.seed()
    neural_network = None
    training_set = { "input" : [], "target" : [] }
    validation_set = { "input" : [], "target" : [] }

    ### Loading word vectors ###
    print("Loading word vectors...")
    with lt.LookupTable() as word2vec:
        word2vec.load(argv[2])

        ### Loading dataset ###
        print("Loading dataset...")
        data = parser.load_dataset(argv[1])

        integer_sentences = [[], []]
        max_sentence_length = max(len(line) for label in data for file in data[label] for line in data[label][file])
        print("Max sentence length is {0}. Sentences will be padded to this size to make the use of batches possible.".format(max_sentence_length))
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

        for i in xrange(cv_reviews):
            sample_class = random.randint(0, len(integer_sentences) - 1)
            sentence = random.randint(0, len(integer_sentences[sample_class]) - 1)
            training_set["input"].append(integer_sentences[sample_class][sentence])
            training_set["target"].append([1.0, 0.0] if sample_class == 0 else [0.0, 1.0])
            del integer_sentences[sample_class][sentence]

        for class_idx, sample_class in enumerate(integer_sentences):
            for sentence in xrange(len(sample_class)):
                validation_set["input"].append(sample_class[sentence])
                validation_set["target"].append([1.0, 0.0] if class_idx == 0 else [0.0, 1.0])

        ### Create Neural network, train it, and save it ###
        print("Creating network...")
        neural_network = nn.NeuralNetwork(number_of_classes = 2,
                                          vector_dimension = word2vec.getWordDimension(),
                                          vocabulary_size = word2vec.getVocabularySize(),
                                          sentence_length = max_sentence_length,
                                          dictionnary = word2vec.getLookupTable())

    ### TESTS ###
    print("------ Accuracy test before training on training set")
    print(neural_network.accuracy(training_set["input"], training_set["target"]))

    print("------ Training network")
    i = 0
    j = 1
    k = 5
    increase_strip = 0
    min_error = None
    validation_error = []
    while True:
        print("Epoch {0}".format(i + 1))

        ### Shuffling training set ###
        tmp = list(zip(training_set["input"], training_set["target"]))
        random.shuffle(tmp)
        training_set["input"], training_set["target"] = zip(*tmp)
        del tmp

        print("Training...")
        neural_network.train(training_set["input"], training_set["target"], 50)
        if (j == k):
            validation_error.append(1.0 - neural_network.accuracy(validation_set["input"], validation_set["target"]))
            if (min_error is None) or (validation_error[-1] < min_error):
                min_error = validation_error[-1]
                neural_network.save("./SAVES/Save")
            if (len(validation_error) > 1 and validation_error[-1] > validation_error[-2]):
                increase_strip += 1
            else:
                increase_strip = 0
            print("Error on training set is {2}. Error on validation set is {0}. Strip of {1} increases in validation error.".format(validation_error[-1], increase_strip, 1.0 - neural_network.accuracy(training_set["input"], training_set["target"])))
            if increase_strip == 3:
                break
            j = 0
        j += 1
        i += 1

    #Re-establish best weight set
    neural_network.restore("./SAVES/Save")

    print("------ Final accuracy tests")
    print("--- Training set")
    print(neural_network.accuracy(training_set["input"], training_set["target"]))
    print("--- CV set")
    print(neural_network.accuracy(validation_set["input"], validation_set["target"]))
    return

if __name__ == "__main__":
    main(sys.argv)
