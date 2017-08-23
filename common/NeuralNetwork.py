
# A class that builds a neural network using the model from Kim Yoon, 2014, using TensorFlow, and provides various utilities to save and explore it #
# ------------------------ #

from sys import stdout

import os
import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, number_of_classes, vector_dimension, sentence_length, vocabulary_size, dictionnary, is_trainable=False, gpu=True):
        ### Network settings ###
        self.vector_dimension = vector_dimension

        ### Placeholders declaration ###
        self.dropout_rate = tf.placeholder(tf.float32)
        self.input_sentence = tf.placeholder(tf.int32, shape = [None, sentence_length])
        self.input_dictionnary = tf.placeholder(tf.float32, shape = [vocabulary_size, vector_dimension])
        self.training_labels = tf.placeholder(tf.float32, shape = [None, number_of_classes])

        ### Creating weights ###
        # Create 100 kernels of 3 * k, 100 kernels of 4 * k, 100 kernels of 5 * k
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

        # Create 300 * 2 weights for the output layer (2 classes : positive + negative)
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

        # Produce feature maps
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
        self.optimiser = tf.train.AdadeltaOptimizer(learning_rate = 1.0)
        losses = tf.nn.softmax_cross_entropy_with_logits(self.output, self.training_labels)
        self.loss = tf.reduce_mean(losses)

        # Ask the optimiser compute gradients, apply l2 constraint, and to apply the gradients
        self.grads_and_vars = self.optimiser.compute_gradients(self.loss, [#self.dictionnary,
            self.kernels[3], self.kernels[4], self.kernels[5],
            self.biases[3], self.biases[4], self.biases[5]])

        self.hidden_layer_grads_and_vars = self.optimiser.compute_gradients(self.loss, [self.hidden_layer_weights, self.hidden_layer_bias])
        self.capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm = 3.0, axes = [0]), gv[1]) for gv in self.hidden_layer_grads_and_vars]

        self.optimisation_step = self.optimiser.apply_gradients(self.grads_and_vars)
        self.optimisation_step2 = self.optimiser.apply_gradients(self.capped_grads_and_vars, global_step = self.global_step)

        ### Accuracy test ###
        correct_predictions = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.training_labels, 1))
        self.accuracy_value = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        ### Saver ###
        self.saver = tf.train.Saver(tf.global_variables() + [self.global_step])

        ## Session and initialisation ##
        if gpu:
            config = tf.ConfigProto(allow_soft_placement = True)
        else:
            config = tf.ConfigProto(allow_soft_placement = True, device_count = {'GPU': 0})
        self.session = tf.Session(config = config)
        self.session.run(tf.global_variables_initializer(), {self.input_dictionnary : dictionnary})
        return

    def train(self, training_set, target_labels, batch_size):
        ### Run optimization ###
        begin = 0
        end = batch_size
        while begin < len(training_set):
            _, _, loss = self.session.run([self.optimisation_step, self.optimisation_step2, self.loss], feed_dict = {
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
    
    def accuracy(self, input, target, subdivision=1500):
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

    def variance_analysis(self, input, subdivision=1500):
        results = []
        begin = 0
        end = subdivision

        summed_squared_activation_3 = tf.reduce_mean(tf.square(self.convolution_3_output), [1, 2])
        summed_squared_activation_4 = tf.reduce_mean(tf.square(self.convolution_4_output), [1, 2])
        summed_squared_activation_5 = tf.reduce_mean(tf.square(self.convolution_5_output), [1, 2])

        summed_activation_3 = tf.reduce_mean(self.convolution_3_output, [1, 2])
        summed_activation_4 = tf.reduce_mean(self.convolution_4_output, [1, 2])
        summed_activation_5 = tf.reduce_mean(self.convolution_5_output, [1, 2])

        Sssa3, Sssa4, Sssa5, Ssa3, Ssa4, Ssa5 = (None, None, None, None, None, None)

        while begin < len(input):
            ssa3, ssa4, ssa5, sa3, sa4, sa5 = self.session.run([
                summed_squared_activation_3,
                summed_squared_activation_4,
                summed_squared_activation_5,
                summed_activation_3,
                summed_activation_4,
                summed_activation_5],
                feed_dict = {
                    self.input_sentence : input[begin:end]
                }
            )

            if begin == 0:
                Sssa3 = ssa3; Sssa4 = ssa4; Sssa5 = ssa5; Ssa3 = sa3; Ssa4 = sa4; Ssa5 = sa5
            else:
                Sssa3 = np.concatenate((Sssa3, ssa3), axis = 0)
                Sssa4 = np.concatenate((Sssa4, ssa4), axis = 0)
                Sssa5 = np.concatenate((Sssa5, ssa5), axis = 0)
                Ssa3 = np.concatenate((Ssa3, sa3), axis = 0)
                Ssa4 = np.concatenate((Ssa4, sa4), axis = 0)
                Ssa5 = np.concatenate((Ssa5, sa5), axis = 0)

            begin += subdivision
            end += subdivision

        #n = tf.constant(len(input), dtype = tf.float32)
        variance_3 = tf.reduce_mean(tf.constant(Sssa3), 0) - tf.square(tf.reduce_mean(tf.constant(Ssa3), 0))
        variance_4 = tf.reduce_mean(tf.constant(Sssa4), 0) - tf.square(tf.reduce_mean(tf.constant(Ssa4), 0))
        variance_5 = tf.reduce_mean(tf.constant(Sssa5), 0) - tf.square(tf.reduce_mean(tf.constant(Ssa5), 0))

        v3, v4, v5 = self.session.run([variance_3, variance_4, variance_5])
        print(v3)
        print(v4)
        print(v5)
        return 0

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

    def create_kernel_maximiser(self, trials, kernel_size, neuron_index):

        maximiser_input = tf.Variable(tf.random_uniform([trials, kernel_size, self.vector_dimension], -1.0, 1.0), trainable = True)
        kernel = tf.slice(self.kernels[kernel_size], [0, 0, 0, neuron_index], [kernel_size, self.vector_dimension, 1, 1])

        output = tf.nn.conv2d(tf.expand_dims(maximiser_input, -1),
            kernel,
            strides = [1, 1, 1, 1],
            padding = "VALID")

        return Maximiser(self.session, maximiser_input, output, self)

class Maximiser:
    def __init__(self, session, input, value_to_optimise, neural_network):

        self.session = session
        self.result = input
        self.value_to_optimise = value_to_optimise
        self.nn = neural_network

        ### Maximisation ###
        self.best_result_values = []

        ### Computation graph for cosine similarity ###
        self.tensor_slice = tf.placeholder(tf.float32, [None, 1, 300])
        batched_tensor_slice = tf.tile(tf.nn.l2_normalize(self.tensor_slice, 2), [1, 50000, 1])
        self.normalized_lookup_table_full = tf.nn.l2_normalize(self.nn.dictionnary, 1)
        self.normalized_lookup_table = tf.placeholder(tf.float32, shape = [50000, self.nn.vector_dimension])

        self.batched_cosine_similarity = tf.reduce_sum(tf.multiply(batched_tensor_slice, tf.expand_dims(self.normalized_lookup_table, 0)), 2)
        self.best_cosine_similarity = tf.nn.top_k(self.batched_cosine_similarity, k = 5)
        self.worst_cosine_similarity = tf.nn.top_k(tf.negative(self.batched_cosine_similarity), k = 5)

        ### Create optimiser ###
        self.optimiser = tf.train.GradientDescentOptimizer(0.01)
        gradients = self.optimiser.compute_gradients(self.value_to_optimise, [self.result])
        self.optimisation_step = self.optimiser.apply_gradients([tf.negative(gv[0]), gv[1]] for gv in gradients)
        self.normalisation_step = tf.assign(input, tf.nn.l2_normalize(input, 1))

        self.session.run(tf.variables_initializer([input]))
        return

    def run(self):
        print("Maximizing output")
        last_values = self.session.run(self.value_to_optimise)
        previous_values = None
        while previous_values is None or sum(abs(last_values - previous_values)) / len(last_values) > 0.000001:
            _, _, self.best_result_values = self.session.run([self.optimisation_step, self.normalisation_step, self.value_to_optimise])
            stdout.write("\r")
            for value in self.best_result_values:
                stdout.write("%f\t" % value)
            stdout.flush()
            previous_values = last_values
            last_values = self.session.run(self.value_to_optimise)
        stdout.write("\n")
        return

    def find_maximising_sentence(self, lookup_table):

        input_shape = self.session.run(tf.shape(self.result))
        words_in_input = input_shape[1]
        sentence = []

        for i in xrange(words_in_input):
            result_slice = self.session.run(tf.slice(self.result, [0, i, 0], [-1, 1, self.nn.vector_dimension]))

            normalized_lookup_table_values = self.session.run(self.normalized_lookup_table_full)

            begin = 0
            end = 50000
            best_words = { "values" : [], "indices" : []}
            worst_words = { "values" : [], "indices" : []}
            for i in xrange(result_slice.shape[0]):
                best_words["values"].append([])
                best_words["indices"].append([])
                worst_words["values"].append([])
                worst_words["indices"].append([])

            while begin < len(normalized_lookup_table_values):
                words_similarity, part_best_words, part_worst_words = self.session.run([self.batched_cosine_similarity, self.best_cosine_similarity, self.worst_cosine_similarity], feed_dict =
                {
                    self.tensor_slice : result_slice,
                    self.normalized_lookup_table : normalized_lookup_table_values[begin:end] if end < len(normalized_lookup_table_values) else np.concatenate([normalized_lookup_table_values[begin:end], np.zeros((50000 - len(normalized_lookup_table_values) + begin, 300))])
                })

                for i in xrange(len(part_best_words.indices)):
                    for j in xrange(part_best_words.indices[i].shape[0]):
                        part_best_words.indices[i][j] += begin
                    best_words["values"][i] += part_best_words.values[i].tolist()
                    best_words["indices"][i] += part_best_words.indices[i].tolist()

                for i in xrange(len(part_worst_words.indices)):
                    for j in xrange(part_worst_words.indices[i].shape[0]):
                        part_worst_words.indices[i][j] += begin
                    worst_words["values"][i] += part_worst_words.values[i].tolist()
                    worst_words["indices"][i] += part_worst_words.indices[i].tolist()

                begin += 50000
                end += 50000

            for i in xrange(len(best_words["values"])):
                best_words["values"][i], best_words["indices"][i] = zip(*(sorted(zip(best_words["values"][i], best_words["indices"][i]), key = lambda element: element[0], reverse=True)))
                worst_words["values"][i], worst_words["indices"][i] = zip(*(sorted(zip(worst_words["values"][i], worst_words["indices"][i]), key = lambda element: element[0], reverse=True)))

            sentence.append([])
            print(len(best_words["values"]))
            for i in xrange(len(best_words["values"])):
                sentence[-1].append([])
                for j in xrange(5): # DEBUG
                    max_word = lookup_table.reverse_lookup(best_words["indices"][i][j])
                    max_diff = best_words["values"][i][j]

                    min_word = lookup_table.reverse_lookup(worst_words["indices"][i][j])
                    min_diff = worst_words["values"][i][j]

                    print("Optimum {4} #{5} : {0} {1} | {2} {3}".format(max_word, max_diff, min_word, min_diff, i, j))
                    sentence[-1][-1].append({"bestword" : max_word, "bestdiff" : max_diff, "worstword" : min_word, "worstdiff" : min_diff})
        return sentence
