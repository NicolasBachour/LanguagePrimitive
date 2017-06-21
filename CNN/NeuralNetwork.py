
# ADD THE BIAS WEIGHT to each kernel
# ADD THE ACTIVATION FUNCTION

import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, vector_dimention):

        self.vector_dimention = vector_dimention

        self.kernels = {}

        # Create 100 kernels of 3 * k
        self.kernels[3] = tf.Variable(self.__shape_to_variables([1, 3 * self.vector_dimention, 100]), tf.float32)

        # Create 100 kernels of 4 * k
        self.kernels[4] = tf.Variable(self.__shape_to_variables([1, 4 * self.vector_dimention, 100]), tf.float32)

        # Create 100 kernels of 5 * k
        self.kernels[5] = tf.Variable(self.__shape_to_variables([1, 5 * self.vector_dimention, 100]), tf.float32)

        # Create 300 * 2 weights for the output layer (2 classes : positive + negative)
        self.hidden_layer_weights = tf.Variable(self.__shape_to_variables([300, 2]), tf.float32)

        self.global_initialiser = tf.global_variables_initializer()
        return

    def run(self, input):
        results = self.__run(input)
        return results # DEBUG
        return results.output
    
    def __run(self, input_data):
        # actually run
        # return lots of details, like error rate (SSE)...

        sentence_length = len(input_data) # refered as n in calculations below

        # If input data is too short, add padding
        while sentence_length < 5:
            input_data.append(np.zeros(self.vector_dimention, dtype=np.float32))
            sentence_length += 1

        convoluted_input_3 = tf.placeholder(tf.float32, shape = [sentence_length - 2, 1, 3 * self.vector_dimention])  # of shape (n - h + 1), 1, hk
        convoluted_input_4 = tf.placeholder(tf.float32, shape = [sentence_length - 3, 1, 4 * self.vector_dimention])  # of shape (n - h + 1), 1, hk
        convoluted_input_5 = tf.placeholder(tf.float32, shape = [sentence_length - 4, 1, 5 * self.vector_dimention])  # of shape (n - h + 1), 1, hk

        # Tile the input matrix in the three "convoluted_input"
        convoluted_input_3_values = self.__map_input(input_data, 3)
        convoluted_input_4_values = self.__map_input(input_data, 4)
        convoluted_input_5_values = self.__map_input(input_data, 5)

        print(convoluted_input_3_values)

        # Produce feature maps (of shape [k - s + 1, 100, 300])
        convolution_3_feature_map = tf.tanh(tf.matmul(convoluted_input_3, tf.tile(self.kernels[3], [sentence_length - 2, 1, 1]))) # of shape [n - h + 1, 1, 100]
        convolution_4_feature_map = tf.tanh(tf.matmul(convoluted_input_4, tf.tile(self.kernels[4], [sentence_length - 3, 1, 1]))) # of shape [n - h + 1, 1, 100]
        convolution_5_feature_map = tf.tanh(tf.matmul(convoluted_input_5, tf.tile(self.kernels[5], [sentence_length - 4, 1, 1]))) # of shape [n - h + 1, 1, 100]

        # Max overtime pooling
        feature_3 = tf.reduce_max(convolution_3_feature_map, reduction_indices = [0])       # of size [1, 100]
        feature_4 = tf.reduce_max(convolution_4_feature_map, reduction_indices = [0])       # of size [1, 100]
        feature_5 = tf.reduce_max(convolution_5_feature_map, reduction_indices = [0])       # of size [1, 100]

        # Output
        # Concatenate along axis 1
        feature_vector = tf.concat(1, [feature_3, feature_4, feature_5]) # of size [1, 300]
        #DEBUG
        output_without_softmax = tf.matmul(feature_vector, self.hidden_layer_weights) # of size [2, 1]        
        output = tf.nn.softmax(output_without_softmax)

        with tf.Session() as session:
            session.run(self.global_initialiser)
            results = session.run(output_without_softmax, feed_dict=
            {
                convoluted_input_3 : convoluted_input_3_values,
                convoluted_input_4 : convoluted_input_4_values,
                convoluted_input_5 : convoluted_input_5_values
            }) #DEBUG
            print(results)
            results = session.run(output, feed_dict=
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
        for i in xrange(0, sentence_length - count + 1):
            mapped_input.append([])
            mapped_input[i].append([])
            for j in xrange(0, count):
                mapped_input[i][0].extend(input[i + j])
        return mapped_input

    def __shape_to_variables(self, shape):
        # DEBUG
        return np.random.random(shape).astype(np.float32)


