
import sys
import os
from common import parser
from common import NeuralNetwork as nn
from common import LookupTable as lt

MAXIMISER_INSTANCES = 9 # Should be 9

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

    use_selection = False
    if "--select" in argv:
        del argv[argv.index("--select")]
        use_selection = True

    if (len(argv) < 4):
        print("Please specify the dataset, the file containing the word2vec dictionnary, and the folder containing the trained weights of the network  :")
        print("main.py <dataset> <word2vec> <save>")
        print("Optionnal arguments :\n--select\t\t-> allows to select the kernel to be maximised.\n--cpu/--gpu (default)\t-> allows to switch computations to CPU/GPU")
        return

    print("Loading word vectors...")
    with lt.LookupTable() as word2vec:
        word2vec.load(argv[2])

        ### Loading dataset ###
        print("Loading dataset...")
        data = parser.load_dataset(argv[1])

        if load_as_CR:
            _, max_sentence_length = load_dataset_as_CR(data, word2vec)
        else:
            _, max_sentence_length = load_dataset_as_MR(data, word2vec)

        print("Creating static network...")
        neural_network = nn.NeuralNetwork(number_of_classes = 2,
                                          vector_dimension = word2vec.getWordDimension(),
                                          vocabulary_size = word2vec.getVocabularySize(),
                                          sentence_length = max_sentence_length,
                                          dictionnary = word2vec.getLookupTable(),
                                          is_trainable = False,
                                          gpu = use_gpu)

        print("Restoring weights...")
        neural_network.restore(os.path.join(argv[3], "Save"))

        #-------------------------

        if use_selection:
            while True:
                x = int(input("Kernel size : "))
                y = int(input("Index : "))
                print("Maximising neuron...")
                maximiser = neural_network.create_kernel_maximiser(MAXIMISER_INSTANCES, x, y)
                maximiser.run()
                sentence = maximiser.find_maximising_sentence(word2vec)
                print(sentence)
        else:
            for x in xrange(4, 6):
                for y in xrange(100):
                    print("Maximising neuron {0}/{1}...".format(x, y))
                    maximiser = neural_network.create_kernel_maximiser(MAXIMISER_INSTANCES, x, y)
                    maximiser.run()
                    sentence = maximiser.find_maximising_sentence(word2vec)
                    print(sentence)
    return

def load_dataset_as_MR(data, word2vec):
    sentence_set = {
        "char" : [],
        "input" : [],
        "target" : []
    }

    max_sentence_length = max(len(line) for label in data for file in data[label] for line in data[label][file])
    print("Max sentence length is {0}. Sentences will be padded to this size to make the use of batches possible.".format(max_sentence_length))

    print("Storing sentences, inputs, and targets...")
    for i, label in enumerate(data):
        for file in data[label]:
            for line in data[label][file]:
                sentence_set["char"].append(line)
                sentence_set["input"].append(word2vec.convertSentence(line, max_sentence_length))
                sentence_set["target"].append([1.0, 0.0] if i == 0 else [0.0, 1.0])
    return sentence_set, max_sentence_length

def load_dataset_as_CR(data, word2vec):
    sentence_set = {
        "char" : [],
        "input" : [],
        "target" : []
    }

    max_sentence_length = max(len(line) for file in data for line in data[file])
    print("Max sentence length is {0}. Sentences will be padded to this size to make the use of batches possible.".format(max_sentence_length))

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
                            sentence_set["char"].append(line)
                            sentence_set["input"].append(word2vec.convertSentence(line, max_sentence_length))
                            sentence_set["target"].append([1.0, 0.0])
                    elif "[-" in line_info:
                        sentence_set["char"].append(line)
                        sentence_set["input"].append(word2vec.convertSentence(line, max_sentence_length))
                        sentence_set["target"].append([0.0, 1.0])
                    break
    return sentence_set, max_sentence_length

if __name__ == "__main__":
    main(sys.argv)
