
import sys
import parser

import numpy as np # DEBUG
import NeuralNetwork as nn

def main(argv):
    """
    if (len(argv) < 2):
        print("Please specify the folder containing the dataset.")
        return
    data = parser.load_dataset(argv[1])
    """

    #for line in content:
        #print("\t" + line)
    #content = [x.strip() for x in content]
    #print("\t" + filename + " in " + folder)

    # Map words in sentences to word vectors




    # Create Neural network, train it, and save it

    neural_network = nn.NeuralNetwork()

    #test = [[0] * 300, [1] * 300, [2] * 300, [3] * 300, [4] * 300, [5] * 300, [6] * 300]
    test = np.random.random([6, 300]).astype(np.float32)
    neural_network.run(test)

    #neural_network.train(vector_input_data) # FUTURE
    #neural_network.save("./SAVES") # FUTURE
    return

if __name__ == "__main__":
    main(sys.argv)
