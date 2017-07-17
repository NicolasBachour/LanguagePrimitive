
import sys
import random
from common import parser
from common import NeuralNetwork as nn
from common import LookupTable as lt

### Constants ###
EARLY_STOPPING__STRIP_LENGTH = 5
CROSS_VALIDATION__FOLD = 10
CROSS_VALIDATION__TEST_COUNT = 1

def main(argv):
    if (len(argv) < 3):
        print("Please specify the folder containing the dataset and the file containing the word2vec dictionnary :")
        print("main.py <dataset> <word2vec>")
        return

    random.seed()
    neural_network = None
    data_subsets = []

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
        for i, label in enumerate(data):
            for file in data[label]:
                for line in data[label][file]:
                    integer_sentences[i].append(word2vec.convertSentence(line, max_sentence_length))

        # Create cv set
        print("Creating cross-validation subset...")
        positive_reviews = len(integer_sentences[0])
        negative_reviews = len(integer_sentences[1])
        cv_reviews = (positive_reviews + negative_reviews) / CROSS_VALIDATION__FOLD
        
        print("\tFound {0} positive and {1} negative reviews...".format(positive_reviews, negative_reviews))
        print("\tSplitting dataset in {0} subsets of {1} reviews each...".format(CROSS_VALIDATION__FOLD, cv_reviews))

        for k in xrange(CROSS_VALIDATION__FOLD):
            data_subsets.append({ "input" : [], "target" : [] })
            for i in xrange(cv_reviews):
                sample_class = random.randint(0, len(integer_sentences) - 1)
                sentence = random.randint(0, len(integer_sentences[sample_class]) - 1)
                data_subsets[k]["input"].append(integer_sentences[sample_class][sentence])
                data_subsets[k]["target"].append([1.0, 0.0] if sample_class == 0 else [0.0, 1.0])
                del integer_sentences[sample_class][sentence]

        ### Create Neural network, train it, and save it ###
        print("Creating network...")
        neural_network = nn.NeuralNetwork(number_of_classes = 2,
                                          vector_dimension = word2vec.getWordDimension(),
                                          vocabulary_size = word2vec.getVocabularySize(),
                                          sentence_length = max_sentence_length,
                                          dictionnary = word2vec.getLookupTable(),
                                          is_trainable = True)

    print("Training network...")
    final_errors = []
    for k in xrange(CROSS_VALIDATION__TEST_COUNT):
        print("==== Starting cross-validation on set {0} ====".format(k))
        validation_set = data_subsets[k]
        training_set = { "input" : [], "target" : [] }
        for i in xrange(CROSS_VALIDATION__FOLD):
            if (i != k):
                training_set["input"].append(data_subsets[i]["input"])
                training_set["target"].append(data_subsets[i]["target"])

        epoch = 0
        increase_strip = 0
        i = 1
        min_error = None
        validation_error = []
        while True:
            print("\tEpoch {0}".format(epoch + 1))
            shuffle_paired_sets(training_set["input"], training_set["target"])
            neural_network.train(training_set["input"], training_set["target"], 50)
            if (i == EARLY_STOPPING__STRIP_LENGTH):
                validation_error.append(1.0 - neural_network.accuracy(validation_set["input"], validation_set["target"]))
                if (min_error is None) or (validation_error[-1] < min_error):
                    min_error = validation_error[-1]
                    neural_network.save("./SAVE_" + k + "/Save")
                if len(validation_error) > 1:
                    if (validation_error[-1] - validation_error[-2]) < 0.1:
                        break
                    if validation_error[-1] > validation_error[-2]:
                        increase_strip += 1
                        if increase_strip == 3:
                            break
                    else:
                        increase_strip = 0
                print("\tError on training set is {2}. Error on validation set is {0}. Strip of {1} increases in validation error.".format(validation_error[-1], increase_strip, 1.0 - neural_network.accuracy(training_set["input"], training_set["target"])))
                i = 0
            i += 1
            epoch += 1

        #Re-establish best weight set
        neural_network.restore("./SAVE_" + k + "/Save")
        final_errors.append(neural_network.accuracy(validation_set["input"], validation_set["target"]))

        print("\tAccuracy test on validation set {0}".format(k))
        print("\tTraining set : {0}    |    Validation set : {1}".format(neural_network.accuracy(training_set["input"], training_set["target"]), final_errors[-1]))

    Final_mean_error = sum(final_errors) / float(len(final_errors))
    print("Final accuracy tests yields a performance of {0} % in accuracy".format(Final_mean_error))
    return

def shuffle_paired_sets(a, b):
    tmp = list(zip(a, b))
    random.shuffle(tmp)
    a, b = zip(*tmp)
    return

if __name__ == "__main__":
    main(sys.argv)
