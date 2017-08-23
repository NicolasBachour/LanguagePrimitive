
# This script trains the neural network using the appropriate datasets and saves its weights for further use in other scripts. #
# The model it uses is similar to that of Kim Yoon, 2014, but rebuilt with TensorFlow for more ease during its exploration.
# - Created by Nicolas Bachour, 2017
# ------------------------ #

import sys
import random
from common import parser
from common import NeuralNetwork as nn
from common import LookupTable as lt

### Constants ###
# Variable determining the number of training epochs which trigger an evaluation of the network's accuracy
EARLY_STOPPING__STRIP_LENGTH = 1
# Number of increases in the validation accuracy of the network which will end the training process
EARLY_STOPPING__INCREASES = 3
# Number of subsets in which the data are split for cross-validation
CROSS_VALIDATION__FOLD = 10
# Number of subsets on which the network final accuracy is evaluated
CROSS_VALIDATION__TEST_COUNT = 1 # The value here is 1 instead of 10, since we do not insist on proving that our network is consistently efficient.

### Program entry point ###
def main(argv):
    use_gpu = None
    load_as_CR = None

    if "--gpu" in argv:
        del argv[argv.index("--gpu")]
        use_gpu = True
    if "--cpu" in argv:
        if not use_gpu is None:
            print("Please specify either the --gpu or --cpu option.")
            return
        del argv[argv.index("--cpu")]
        use_gpu = False
    if use_gpu == None:
        use_gpu = True

    if "--CR" in argv:
        del argv[argv.index("--CR")]
        load_as_CR = True
    if "--MR" in argv:
        if not load_as_CR is None:
            print("Please specify either the --CR or --MR option.")
            return
        del argv[argv.index("--MR")]
        load_as_CR = False
    if load_as_CR == None:
        print("Please specify either the --CR or --MR option.")
        return
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
        if load_as_CR:
            integer_sentences, max_sentence_length = load_dataset_as_CR(data, word2vec)
        else:
            integer_sentences, max_sentence_length = load_dataset_as_MR(data, word2vec)

        # Create CV sets
        for k in xrange(CROSS_VALIDATION__FOLD):
            data_subsets.append({ "input" : [], "target" : [] })
            for class_idx in xrange(2):
                for sentence in integer_sentences[class_idx][k]:
                    data_subsets[k]["input"].append(sentence)
                    data_subsets[k]["target"].append([1.0, 0.0] if class_idx == 0 else [0.0, 1.0])

        ### Create Neural network, train it, and save it ###
        print("Creating network...")
        neural_network = nn.NeuralNetwork(number_of_classes = 2,
                                          vector_dimension = word2vec.getWordDimension(),
                                          vocabulary_size = word2vec.getVocabularySize(),
                                          sentence_length = max_sentence_length,
                                          dictionnary = word2vec.getLookupTable(),
                                          is_trainable = True,
                                          gpu = use_gpu)

    print("Training network...")
    final_errors = []
    for k in xrange(CROSS_VALIDATION__TEST_COUNT):
        print("==== Starting cross-validation on set {0} ====".format(k))
        validation_set = data_subsets[k]
        training_set = { "input" : [], "target" : [] }
        for i in xrange(CROSS_VALIDATION__FOLD):
            if (i != k):
                training_set["input"] += data_subsets[i]["input"]
                training_set["target"] += data_subsets[i]["target"]
        
        epoch = 0
        increase_strip = 0
        i = 1
        min_error = None
        validation_error = []
        while True:
            print("\tEpoch {0}".format(epoch + 1))
            training_set["input"], training_set["target"] = shuffle_paired_sets(training_set["input"], training_set["target"])
            neural_network.train(training_set["input"], training_set["target"], 50)
            if (i == EARLY_STOPPING__STRIP_LENGTH):
                validation_error.append(1.0 - neural_network.accuracy(validation_set["input"], validation_set["target"]))
                if (min_error is None) or (validation_error[-1] < min_error):
                    min_error = validation_error[-1]
                    neural_network.save("./SAVE_" + str(k) + "/Save")
                if len(validation_error) > 1:
                    if validation_error[-1] > validation_error[-2]:
                        increase_strip += 1
                        if increase_strip == EARLY_STOPPING__INCREASES:
                            break
                    else:
                        increase_strip = 0
                print("\tError on training set is {2}. Error on validation set is {0}. Strip of {1} increases in validation error.".format(validation_error[-1], increase_strip, 1.0 - neural_network.accuracy(training_set["input"], training_set["target"])))
                i = 0
            i += 1
            epoch += 1

        #Re-establish best weight set
        neural_network.restore("./SAVE_" + str(k) + "/Save")
        final_errors.append(neural_network.accuracy(validation_set["input"], validation_set["target"]))

        print("\tAccuracy test on validation set {0}".format(k))
        print("\tTraining set : {0}    |    Validation set : {1}".format(neural_network.accuracy(training_set["input"], training_set["target"]), final_errors[-1]))

    Final_mean_error = sum(final_errors) / float(len(final_errors))
    print("Final accuracy tests yields a performance of {0} % in accuracy".format(Final_mean_error))
    return

def shuffle_paired_sets(a, b):
    tmp = list(zip(a, b))
    random.shuffle(tmp)
    return zip(*tmp)

def load_dataset_as_MR(data, word2vec):

    integer_sentences = [[], []]
    max_sentence_length = max(len(line) for label in data for file in data[label] for line in data[label][file])
    print("Max sentence length is {0}. Sentences will be padded to this size to make the use of batches possible.".format(max_sentence_length))
    for i, label in enumerate(data):
        for k in xrange(CROSS_VALIDATION__FOLD):
            integer_sentences[i].append([])
        for file in data[label]:
            cv_set = int(file[2:5]) / 100
            for line in data[label][file]:
                integer_sentences[i][cv_set].append(word2vec.convertSentence(line, max_sentence_length))
    return integer_sentences, max_sentence_length

def load_dataset_as_CR(data, word2vec):

    integer_sentences = [[], []]
    max_sentence_length = max(len(line) for file in data for line in data[file])
    for i in xrange(2):
        for k in xrange(CROSS_VALIDATION__FOLD):
            integer_sentences[i].append([])
    for file in data:
        for line in data[file]:
            for k in xrange(len(line)):
                if "##" in line[k]:
                    line_strip = line[k].split("##")
                    line_info = line_strip[0]
                    line[k] = line_strip[1]
                    for n in xrange(k):
                        del line[0]
                    if "[+" in line_info:
                        if not "[-" in line_info:
                            integer_sentences[0][0].append(word2vec.convertSentence(line, max_sentence_length))
                    elif "[-" in line_info:
                        integer_sentences[1][0].append(word2vec.convertSentence(line, max_sentence_length))
                    break

    # Split data into random subsets
    print("Splitting reviews in {0} random subsets".format(CROSS_VALIDATION__FOLD))
    for i in xrange(2):
        for k in xrange(1, CROSS_VALIDATION__FOLD):
            for n in xrange(len(integer_sentences[i][0]) / CROSS_VALIDATION__FOLD):
                elem_index = random.randint(0, len(integer_sentences[i][0]) - 1)
                integer_sentences[i][k].append(integer_sentences[i][0][elem_index])
                del integer_sentences[i][0][elem_index]
    return integer_sentences, max_sentence_length

if __name__ == "__main__":
    main(sys.argv)
