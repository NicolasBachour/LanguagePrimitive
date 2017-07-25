
from sys import stdout

import os
import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, number_of_classes, vector_dimension, sentence_length, vocabulary_size, dictionnary, is_trainable=False):
        ### Network settings ###
        self.vector_dimension = vector_dimension

        ### Placeholders declaration ###
        self.dropout_rate = tf.placeholder(tf.float32)
        self.input_sentence = tf.placeholder(tf.int32, shape = [None, sentence_length])
        self.input_dictionnary = tf.placeholder(tf.float32, shape = [vocabulary_size, vector_dimension])
        self.training_labels = tf.placeholder(tf.float32, shape = [None, number_of_classes])

        ### Creating weights ###
        # Create 100 kernels of 3 * k, 100 kernels of 4 * k, 100 kernels of 5 *
        # k
        self.kernels = {
            3 : tf.Variable(self.__shape_to_variables([3, self.vector_dimension, 1, 100]), trainable = is_trainable),
            4 : tf.Variable(self.__shape_to_variables([4, self.vector_dimension, 1, 100]), trainable = is_trainable),
            5 : tf.Variable(self.__shape_to_variables([5, self.vector_dimension, 1, 100]), trainable = is_trainable)
        }

        self.biases = {
            3 : tf.Variable(self.__shape_to_variables([100]), trainable = is_trainable),
            4 : tf.Variable(self.__shape_to_variables([100]), trainable = is_trainable),
            5 : tf.Variable(self.__shape_to_variables([100]), trainable = is_trainable)
        }

        # Create 300 * 2 weights for the output layer (2 classes : positive +
        # negative)
        self.hidden_layer_weights = tf.Variable(self.__shape_to_variables([300, number_of_classes]), trainable = is_trainable)
        self.hidden_layer_bias = tf.Variable(self.__shape_to_variables([number_of_classes]), trainable = is_trainable)

        ### Build network graph ###
        self.global_step = tf.Variable(0, trainable = False)
        self.dictionnary = tf.Variable(self.input_dictionnary, trainable = False)
        self.embedded_chars = tf.nn.embedding_lookup(self.dictionnary, self.input_sentence)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        self.convolution_3_output = tf.nn.conv2d(self.embedded_chars_expanded,
            self.kernels[3],
            strides = [1, 1, 1, 1],
            padding = "VALID")
        self.convolution_4_output = tf.nn.conv2d(self.embedded_chars_expanded,
            self.kernels[4],
            strides = [1, 1, 1, 1],
            padding = "VALID")
        self.convolution_5_output = tf.nn.conv2d(self.embedded_chars_expanded,
            self.kernels[5],
            strides = [1, 1, 1, 1],
            padding = "VALID")

        # Produce feature maps (of shape [k - s + 1, 100, 300])
        convolution_3_feature_map = tf.nn.relu(tf.nn.bias_add(self.convolution_3_output, self.biases[3])) # of shape [batches, n - h + 1, 1, 100]
        convolution_4_feature_map = tf.nn.relu(tf.nn.bias_add(self.convolution_4_output, self.biases[4])) # of shape [batches, n - h + 1, 1, 100]
        convolution_5_feature_map = tf.nn.relu(tf.nn.bias_add(self.convolution_5_output, self.biases[5])) # of shape [batches, n - h + 1, 1, 100]

        # Max overtime pooling
        feature_3 = tf.nn.max_pool(convolution_3_feature_map, ksize = [1, sentence_length - 2, 1, 1], strides = [1, 1, 1, 1], padding='VALID')
        feature_4 = tf.nn.max_pool(convolution_4_feature_map, ksize = [1, sentence_length - 3, 1, 1], strides = [1, 1, 1, 1], padding='VALID')
        feature_5 = tf.nn.max_pool(convolution_5_feature_map, ksize = [1, sentence_length - 4, 1, 1], strides = [1, 1, 1, 1], padding='VALID')

        # Concatenate to form batch_size * 300
        feature_vector = tf.reshape(tf.concat(3, [feature_3, feature_4, feature_5]), [-1, 300]) # of size [batches, 300]
        self.output = tf.nn.xw_plus_b(feature_vector, tf.nn.dropout(self.hidden_layer_weights, self.dropout_rate), self.hidden_layer_bias)

        ####### Training ######
        self.optimiser = tf.train.AdadeltaOptimizer(learning_rate = 5.0)
        losses = tf.nn.softmax_cross_entropy_with_logits(self.output, self.training_labels)
        self.loss = tf.reduce_mean(losses)

        # Ask the optimiser compute gradients, apply l2 constraint, and to
        # apply the gradients
        self.grads_and_vars = self.optimiser.compute_gradients(self.loss, [#self.dictionnary,
            self.kernels[3], self.kernels[4], self.kernels[5],
            self.biases[3], self.biases[4], self.biases[5],
            self.hidden_layer_weights, self.hidden_layer_bias])
        self.capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm = 3.0, axes = None), gv[1]) for gv in self.grads_and_vars] # TRUC LOUCHE
        self.optimisation_step = self.optimiser.apply_gradients(self.capped_grads_and_vars, global_step = self.global_step)

        ### Accuracy test ###
        correct_predictions = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.training_labels, 1))
        self.accuracy_value = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        ### Saver ###
        self.saver = tf.train.Saver(tf.global_variables() + [self.global_step])

        ## Session and initialisation ##
        #self.session = tf.Session(config = tf.ConfigProto(device_count = {'GPU': 0})) #allow_soft_placement = True, log_device_placement = True
        self.session = tf.Session() #allow_soft_placement = True, log_device_placement = True
        self.session.run(tf.global_variables_initializer(), {self.input_dictionnary : dictionnary})
        return

    def train(self, training_set, target_labels, batch_size):
        ### Run optimization ###
        begin = 0
        end = batch_size
        while begin < len(training_set):
            _, loss = self.session.run([self.optimisation_step, self.loss], feed_dict = {
                self.dropout_rate : 0.5,
                self.input_sentence : training_set[begin:end],
                self.training_labels : target_labels[begin:end]
            })

            stdout.write("\r%f" % loss)
            stdout.flush()
            begin += batch_size
            end += batch_size
        stdout.write("\n")
        return
    
    def accuracy(self, input, target, subdivision=500):
        results = []
        begin = 0
        end = subdivision
        while begin < len(input):
            results.append(self.session.run(self.accuracy_value, feed_dict = {
                self.dropout_rate : 1.0,
                self.input_sentence : input[begin:end],
                self.training_labels : target[begin:end]
            }))
            begin += subdivision
            end += subdivision
        return sum(results) / len(results)

    def save(self, destination):
        folder = os.path.abspath(destination)
        # Create directory if not already existing
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.saver.save(self.session, folder)
        return

    def restore(self, destination):
        folder = os.path.abspath(destination)
        self.saver.restore(self.session, folder)
        return

    ########################
    ### HELPER FUNCTIONS ###
    ########################

    def __shape_to_variables(self, shape):
        return tf.truncated_normal(shape, stddev = 0.1)

    ##############################
    ### MAXIMIZATION FUNCTIONS ###
    ##############################

    def create_kernel_maximiser(self, kernel_size, neuron_index):

        #maximiser_input = tf.Variable(tf.truncated_normal([1, kernel_size, self.vector_dimension, 1], stddev = 1.0), trainable = True)
        maximiser_input = tf.Variable(tf.truncated_normal([kernel_size, self.vector_dimension], stddev = 1.0), trainable = True)
        kernel = tf.slice(self.kernels[kernel_size], [0, 0, 0, neuron_index], [kernel_size, self.vector_dimension, 1, 1])

        output = tf.nn.conv2d( tf.expand_dims(tf.expand_dims(maximiser_input, 0), -1),
            kernel,
            strides = [1, 1, 1, 1],
            padding = "VALID")
        # Without relu layer : useless
        return Maximiser(self.session, maximiser_input, output, self)

class Maximiser:
    def __init__(self, session, input, value_to_optimise, neural_network):

        self.session = session
        self.result = input
        self.value_to_optimise = value_to_optimise
        self.nn = neural_network

        ### Computation graph for cosine similarity ###
        #self.cosine_similarity_operand_1 = tf.placeholder(tf.float32)
        #self.cosine_similarity_operand_2 = tf.placeholder(tf.float32)
        #self.cosine_similarity = tf.reduce_sum(tf.nn.l2_normalize(self.cosine_similarity_operand_1, 0) * tf.nn.l2_normalize(self.cosine_similarity_operand_2, 0))

        self.tensor_slice = tf.placeholder(tf.float32, [1, 300])
        batched_tensor_slice = tf.tile(tf.nn.l2_normalize(self.tensor_slice, 1), [self.session.run(tf.shape(self.nn.dictionnary))[0], 1])
        normalized_lookup_table = tf.nn.l2_normalize(self.nn.dictionnary, 1)
        self.batched_cosine_similarity = tf.reduce_sum(tf.multiply(batched_tensor_slice, normalized_lookup_table), 1)
        self.best_cosine_similarity = tf.arg_max(self.batched_cosine_similarity, 0)

        ### Create optimiser ###
        self.optimiser = tf.train.GradientDescentOptimizer(1.0)
        #self.ratio = tf.div(tf.constant(1.0, tf.float32), self.value_to_optimise)
        #self.optimisation_step = self.optimiser.minimize(self.ratio)
        gradients = self.optimiser.compute_gradients(self.value_to_optimise, [self.result])
        self.optimisation_step = self.optimiser.apply_gradients([tf.negative(gv[0]), gv[1]] for gv in gradients)

        self.session.run(tf.variables_initializer([input]))
        return

    def run(self):
        #print(self.session.run(self.result))
        for i in xrange(3500):
            _, value = self.session.run([self.optimisation_step, self.value_to_optimise])
            stdout.write("\r%f  " % value)
            stdout.flush()
        stdout.write("\n")
        print(self.session.run(self.result))
        return

    def find_maximising_sentence(self, lookup_table):

        input_shape = self.session.run(tf.shape(self.result))
        words_in_input = input_shape[0]
        sentence = []

        for i in xrange(words_in_input):
            result_slice = self.session.run(tf.slice(self.result, [i, 0], [1, self.nn.vector_dimension]))

            words_similarity, best_word_idx = self.session.run([self.batched_cosine_similarity, self.best_cosine_similarity], feed_dict = { self.tensor_slice : result_slice })
            print(best_word_idx)

            min_word = lookup_table.word_to_int.keys()[lookup_table.word_to_int.values().index(best_word_idx)] #lookup_table.lookup_int(best_word_idx) # self.session.run(tf.squeeze(embedded_chars), feed_dict = {self.input_sentence : [best_word_idx]})
            min_diff = words_similarity[int(best_word_idx)]

            #min_diff = None
            #min_word = None
            #for entry in lookup_table.word_to_int:
            #    word = lookup_table.lookup(entry)
            #    diff = self.find_difference(word, tensor_slice)
            #    if min_diff is None or diff < min_diff:
            #        min_diff = diff
            #        min_word = entry
            #        print("Found better solution : {0}".format(entry))
            print("Done : {0} {1}".format(min_word, min_diff))
            sentence.append({"word" : min_word, "diff" : min_diff})
        return sentence

#    def find_difference(self, a, b):
#        return self.__cosine_similarity(a, b)
#
#    def __cosine_similarity(self, a, b):
#        return self.session.run(self.cosine_similarity, feed_dict = 
#        {
#            self.cosine_similarity_operand_1 : a, self.cosine_similarity_operand_2 : b
#        })