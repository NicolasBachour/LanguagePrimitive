
import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self):

        self.kernels = {}

        # Create 100 kernels of 3 * 300
        self.kernels[3] = tf.Variable(self.__shape_to_variables([300, 3, 100]), tf.float32)

        # Create 100 kernels of 4 * 300
        self.kernels[4] = tf.Variable(self.__shape_to_variables([300, 4, 100]), tf.float32)

        # Create 100 kernels of 5 * 300
        self.kernels[5] = tf.Variable(self.__shape_to_variables([300, 5, 100]), tf.float32)


        # Create 300 * 2 weights for the output layer (positive + negative)
        self.hidden_layer_weights = tf.Variable(self.__shape_to_variables([300, 2]), tf.float32)

        # Output matrix is 300

        # Each of the 300 kernels is applied to every word group.......
        # 1 kernel of size s is applied to a sentence of k words, (k - s + 1) times
        # 100 * (k - 2) + 100 * (k - 3) + 100 * (k - 4)
        # (100k - 200) + (100k - 300) + (100k - 400)

        #self.input = tf.placeholder(tf.float32, shape = [20, 300]) # For twenty words.
        #self.output_1 = 

        #self.output = []
        self.global_initialiser = tf.global_variables_initializer()

        return

    def run(self, input):
        results = self.__run(input)
        return results # DEBUG
        return results.output
    
    def __run(self, input_data):
        # actually run
        # return lots of details, like error rate (SSE)...

        sentence_length = len(input_data) # refered as k in calculations below

        convoluted_input_3 = tf.placeholder(tf.float32, shape = [300, sentence_length - 2, 3], name = "trigrams")   # of shape 300, (k - s + 1), s
        convoluted_input_4 = tf.placeholder(tf.float32, shape = [300, sentence_length - 3, 4], name = "tetragrams") # of shape 300, (k - s + 1), s
        convoluted_input_5 = tf.placeholder(tf.float32, shape = [300, sentence_length - 4, 5], name = "pentagrams") # of shape 300, (k - s + 1), s

        # Tile the input matrix in the three "convoluted_input"
        convoluted_input_3_values = np.random.random([300, sentence_length - 2, 3]).astype(np.float32) #self.__map_input(input_data, 3)
        convoluted_input_4_values = np.random.random([300, sentence_length - 3, 4]).astype(np.float32) #self.__map_input(input_data, 4)
        convoluted_input_5_values = np.random.random([300, sentence_length - 4, 5]).astype(np.float32) #self.__map_input(input_data, 5)

        # Produce feature maps (of shape [k - s + 1, 100, 300])
        convolution_3_feature_map = tf.matmul(convoluted_input_3, self.kernels[3])          # of shape [300, k - 2, 100]
        convolution_4_feature_map = tf.matmul(convoluted_input_4, self.kernels[4])          # of shape [300, k - 3, 100]
        convolution_5_feature_map = tf.matmul(convoluted_input_5, self.kernels[5])          # of shape [300, k - 4, 100]

        feature_3 = tf.reduce_max(convolution_3_feature_map, reduction_indices = [1])       # of size [300, 1, 100]
        feature_4 = tf.reduce_max(convolution_4_feature_map, reduction_indices = [1])       # of size [300, 1, 100]
        feature_5 = tf.reduce_max(convolution_5_feature_map, reduction_indices = [1])       # of size [300, 1, 100]

        # Concatenate alog axis 1
        feature_vector = tf.concat(1, [feature_3, feature_4, feature_5]) # of size [300, 1, 300]

        with tf.Session() as session:
            session.run(self.global_initialiser)
            results = session.run(feature_vector, feed_dict=
            {
                convoluted_input_3 : convoluted_input_3_values,
                convoluted_input_4 : convoluted_input_4_values,
                convoluted_input_5 : convoluted_input_5_values
            })
            print(results) # DEBUG
        return results # DEBUG

    def train(self, input_data):
        # train with the cv sample

        return
    
    def save(destination):
        # open file and save as a format as raw as possible
        return

    ########################
    ### HELPER FUNCTIONS ###
    ########################

    def __map_input(self, input, count):
        mapped_input = []
        sentence_length = len(input)
        for i in range(0, count):
            mapped_input.append([])
            for j in range(0, sentence_length - count + 1):
                mapped_input[i].append(input[i + j])
        return mapped_input

    def __shape_to_variables(self, shape):
        # DEBUG
        return np.random.random(shape).astype(np.float32)


