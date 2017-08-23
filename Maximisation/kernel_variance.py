
# This script computes the variance of the activation of the kernels, aiming to establish which kernles are the most representative. #
# ------------------------ #

import sys
import os
from common import parser
from common import NeuralNetwork as nn
from common import LookupTable as lt
import main

### Program entry point ###
def kernel_variance(argv):
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
            input_data, max_sentence_length = main.load_dataset_as_CR(data, word2vec)
        else:
            input_data, max_sentence_length = main.load_dataset_as_MR(data, word2vec)

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

        neural_network.variance_analysis(input_data["input"])

    return

if __name__ == "__main__":
    kernel_variance(sys.argv)
