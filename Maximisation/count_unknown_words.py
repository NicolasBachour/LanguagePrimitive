
# This script computes basic statistics about the number of words present in the dataset and the word2vec dictionnary. #
# ------------------------ #

import sys
import os
from common import parser
from common import LookupTable as lt
import main

### Program entry point ###
def count_main(argv):
    load_as_CR = None
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
        print("Please specify the dataset, and the file containing the word2vec dictionnary :")
        print("main.py <dataset> <word2vec>")
        print("Other arguments :\n--CR\t\t-> Load as the CR (Customer Review) dataset.\n--MR\t\t-> Load as the MR (Movie Review) dataset")
        return

    print("Loading word vectors...")
    with lt.LookupTable() as word2vec:
        word2vec.load(argv[2])

        base_vocabulary_size = word2vec.getVocabularySize()
        print("\tLoaded {0} words from the word2vec dataset.".format(base_vocabulary_size))

        ### Loading dataset ###
        print("Loading dataset...")
        data = parser.load_dataset(argv[1])
        if load_as_CR:
            _, max_sentence_length = main.load_dataset_as_CR(data, word2vec)
        else:
            _, max_sentence_length = main.load_dataset_as_MR(data, word2vec)

        extended_vocabulary_size = word2vec.getVocabularySize()
        print("\tDictionnary size is now of {0} words".format(extended_vocabulary_size))

        print("\tBase\t|\tExtended\t|\tRatio\t|\tIncrease")
        print("\t{0}\t|\t{1}\t|\t{2}%\t|\t{3}%".format(
            base_vocabulary_size,
            extended_vocabulary_size,
            float(base_vocabulary_size) / float(extended_vocabulary_size) * 100.0,
            float(extended_vocabulary_size) / float(base_vocabulary_size) * 100.0,
        ))

    return

if __name__ == "__main__":
    count_main(sys.argv)
