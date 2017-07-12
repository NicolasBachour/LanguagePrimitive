
from sys import stdout

import os
import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, number_of_classes, vector_dimension, sentence_length, vocabulary_size, dictionnary):
        ### Network settings ###
        self.vector_dimension = vector_dimension

        ### Creating weights ###
        # Create 100 kernels of 3 * k, 100 kernels of 4 * k, 100 kernels of 5 * k
        self.kernels = {
            3 : tf.Variable(self.__shape_to_variables([3, self.vector_dimension, 1, 100])),
            4 : tf.Variable(self.__shape_to_variables([4, self.vector_dimension, 1, 100])),
            5 : tf.Variable(self.__shape_to_variables([5, self.vector_dimension, 1, 100]))
        }

        self.biases = {
            3 : tf.Variable(tf.constant(0.1, shape = [100])),
            4 : tf.Variable(tf.constant(0.1, shape = [100])),
            5 : tf.Variable(tf.constant(0.1, shape = [100]))
        }

        # Create 300 * 2 weights for the output layer (2 classes : positive + negative)
        self.hidden_layer_weights = tf.Variable(self.__shape_to_variables([300, number_of_classes]))
        self.hidden_layer_biases = tf.Variable(tf.constant(0.1, shape = [number_of_classes]))
        
        self.dropout_rate = tf.placeholder(tf.float32)

        ### Build network graph ###
        self.input_sentence = tf.placeholder(tf.int32, shape = [None, sentence_length])
        self.input_dictionnary = tf.placeholder(tf.float32, shape = [vocabulary_size, vector_dimension])

        #with tf.device('/cpu:0'), tf.name_scope("embedding"):
        self.dictionnary = tf.Variable(self.input_dictionnary, trainable = False)
        self.embedded_chars = tf.nn.embedding_lookup(self.dictionnary, self.input_sentence)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        self.convolution_3_output = tf.nn.conv2d(
            self.embedded_chars_expanded,
            self.kernels[3],
            strides = [1, 1, 1, 1],
            padding = "VALID")
        self.convolution_4_output = tf.nn.conv2d(
            self.embedded_chars_expanded,
            self.kernels[4],
            strides = [1, 1, 1, 1],
            padding = "VALID")
        self.convolution_5_output = tf.nn.conv2d(
            self.embedded_chars_expanded,
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

        # Output
        # Concatenate to form batch_size * 300
        feature_vector = tf.reshape(tf.concat(3, [feature_3, feature_4, feature_5]), [-1, 300]) # of size [batches, 300]
        self.output = tf.nn.xw_plus_b(tf.nn.dropout(feature_vector, self.dropout_rate), self.hidden_layer_weights, self.hidden_layer_biases)

        ####### Training ######
        self.optimiser = tf.train.AdadeltaOptimizer(learning_rate = 1.0)
        #self.optimiser = tf.train.AdamOptimizer(learning_rate = 1e-4)
        self.training_labels = tf.placeholder(tf.float32, shape = [None, number_of_classes])

        losses = tf.nn.softmax_cross_entropy_with_logits(self.output, self.training_labels)
        self.loss = tf.reduce_mean(losses)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.grads_and_vars = self.optimiser.compute_gradients(self.loss)

        #grads_and_vars = optimizer.compute_gradients(loss, [weights1, weights2, ...])
        self.capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm = 3.0, axes = None), gv[1]) for gv in self.grads_and_vars]
        # Ask the optimizer to apply the capped gradients
        self.optimisation_step = self.optimiser.apply_gradients(self.capped_grads_and_vars, global_step=self.global_step)

        #Accuracy test
        correct_predictions = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.training_labels, 1))
        self.accuracy_value = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        self.saver = tf.train.Saver(tf.global_variables())

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer(), {self.input_dictionnary : dictionnary})
        return

    def train(self, training_set, target_labels, batch_size):

        ### RUN OPTIMISATION ###
        begin = 0
        end = batch_size
        while begin < len(training_set):
            _, loss = self.session.run([self.optimisation_step, self.loss], feed_dict = {
                self.dropout_rate : 1.0,
                self.training_labels : target_labels[begin:end],
                self.input_sentence : training_set[begin:end]
            })

            stdout.write("\r%f" % loss)
            stdout.flush()
            begin += batch_size
            end += batch_size
        stdout.write("\n")
        return
    
    def accuracy(self, input, target, subdivision = 500):
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
