
# ADD THE BIAS WEIGHT to each kernel
# ADD THE ACTIVATION FUNCTION

import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, number_of_classes, vector_dimension, sentence_length, vocabulary_size, dictionnary):
        ### Network settings ###
        self.vector_dimension = vector_dimension

        ### Creating weights ###
        # Create 100 kernels of 3 * k, 100 kernels of 4 * k, 100 kernels of 5 * k
        self.kernels = {
            3 : tf.Variable(self.__shape_to_variables([3, self.vector_dimension, 1, 100]), tf.float32),
            4 : tf.Variable(self.__shape_to_variables([4, self.vector_dimension, 1, 100]), tf.float32),
            5 : tf.Variable(self.__shape_to_variables([5, self.vector_dimension, 1, 100]), tf.float32)
        }

        self.biases = {
            3 : tf.Variable(tf.constant(0.1, shape = [100]), tf.float32),
            4 : tf.Variable(tf.constant(0.1, shape = [100]), tf.float32),
            5 : tf.Variable(tf.constant(0.1, shape = [100]), tf.float32)
        }

        # Create 300 * 2 weights for the output layer (2 classes : positive + negative)
        self.hidden_layer_weights = tf.Variable(self.__shape_to_variables([300, number_of_classes]), tf.float32)
        self.hidden_layer_biases = tf.Variable(tf.constant(0.1, shape = [number_of_classes]))

        ### Build network graph ###
        #self.convoluted_input_3 = tf.placeholder(tf.float32, shape = [None, None, 1, 3 * self.vector_dimension]) # of shape batches, (n - h + 1), 1, hk
        #self.convoluted_input_4 = tf.placeholder(tf.float32, shape = [None, None, 1, 4 * self.vector_dimension]) # of shape batches, (n - h + 1), 1, hk
        #self.convoluted_input_5 = tf.placeholder(tf.float32, shape = [None, None, 1, 5 * self.vector_dimension]) # of shape batches, (n - h + 1), 1, hk

        self.input_sentence = tf.placeholder(tf.int32, shape = [None, sentence_length])
        self.input_dictionnary = tf.placeholder(tf.float32, shape = [vocabulary_size, vector_dimension])

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.dictionnary = tf.Variable(self.input_dictionnary, tf.float32)
            self.embedded_chars = tf.nn.embedding_lookup(self.dictionnary, self.input_sentence)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        #self.conv_output_3 = tf.matmul(self.convoluted_input_3, tf.tile(tf.expand_dims(tf.expand_dims(self.kernels[3], 0), 0), [tf.shape(self.convoluted_input_3)[0], tf.shape(self.convoluted_input_3)[1], 1, 1]))
        #self.conv_output_4 = tf.matmul(self.convoluted_input_4, tf.tile(tf.expand_dims(tf.expand_dims(self.kernels[4], 0), 0), [tf.shape(self.convoluted_input_4)[0], tf.shape(self.convoluted_input_4)[1], 1, 1]))
        #self.conv_output_5 = tf.matmul(self.convoluted_input_5, tf.tile(tf.expand_dims(tf.expand_dims(self.kernels[5], 0), 0), [tf.shape(self.convoluted_input_5)[0], tf.shape(self.convoluted_input_5)[1], 1, 1]))

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
#        feature_3 = tf.reduce_max(convolution_3_feature_map, reduction_indices = [1]) # of size [batches, 1, 100]
#        feature_4 = tf.reduce_max(convolution_4_feature_map, reduction_indices = [1]) # of size [batches, 1, 100]
#        feature_5 = tf.reduce_max(convolution_5_feature_map, reduction_indices = [1]) # of size [batches, 1, 100]
        # DEBUG : Sees tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')    for alternative

        feature_3 = tf.nn.max_pool(convolution_3_feature_map, ksize = [1, sentence_length - 2, 1, 1], strides = [1, 1, 1, 1], padding='VALID')
        feature_4 = tf.nn.max_pool(convolution_4_feature_map, ksize = [1, sentence_length - 3, 1, 1], strides = [1, 1, 1, 1], padding='VALID')
        feature_5 = tf.nn.max_pool(convolution_5_feature_map, ksize = [1, sentence_length - 4, 1, 1], strides = [1, 1, 1, 1], padding='VALID')

        # Output
        # Concatenate to form batch_size * 300
        feature_vector = tf.reshape(tf.concat(3, [feature_3, feature_4, feature_5]), [-1, 300]) # of size [batches, 300]
        #DROPOUT : self.output_with_dropout = tf.nn.softmax(tf.matmul(tf.nn.dropout(feature_vector, 0.5), tf.tile(tf.expand_dims(self.hidden_layer_weights, 0), [tf.shape(self.convoluted_input_3)[0], 1, 1]))) # of size [batches, 2, 1]

        self.output_with_dropout = tf.nn.xw_plus_b(tf.nn.dropout(feature_vector, 0.5), self.hidden_layer_weights, self.hidden_layer_biases)
        self.output = tf.nn.xw_plus_b(feature_vector, self.hidden_layer_weights, self.hidden_layer_biases)

        #self.output_with_dropout = tf.nn.softmax(tf.matmul(feature_vector, tf.tile(tf.expand_dims(self.hidden_layer_weights, 0), [tf.shape(feature_vector)[0], 1, 1]))) # of size [batches, 2, 1]
        #self.output = tf.nn.softmax(tf.matmul(feature_vector, tf.tile(tf.expand_dims(self.hidden_layer_weights, 0), [tf.shape(feature_vector)[0], 1, 1]))) # of size [batches, 2, 1]

        ####### Training ######

        self.optimiser = tf.train.AdadeltaOptimizer(learning_rate = 1.0)

        # training_set is a table containing inputs
        # target_labels is a table containing target outputs
        #number_of_classes = len(target_labels[0]) # number of classes
        # batch_size : number of trials
        # target_labels : vector of length N with values corresponding to the correct class
        # Hand down the weights update to Adadelta
        #indicators = np.zeros((batch_size, number_of_classes))
        #indicators[np.arange(batch_size), target_labels] = 1
        #output_matrix = []
        #for i in xrange(0, batch_size):
        #    output_matrix.append(self.output())

        #loss_convolution_truc = tf.nn.sigmoid_cross_entropy_with_logits

        self.training_labels = tf.placeholder(tf.float32, shape = [None, number_of_classes])
        #self.target_labels = tf.placeholder(tf.float32, shape = [None, 1, number_of_classes])

        #self.loss_hidden_layer = tf.reduce_mean(- tf.reduce_sum(self.training_labels * tf.log(self.output + 1e-10), reduction_indices = [1]))
        #self.loss = tf.reduce_mean(self.loss_hidden_layer)

        losses = tf.nn.softmax_cross_entropy_with_logits(self.output_with_dropout, self.training_labels)
        self.loss = tf.reduce_mean(losses)

        #loss_hidden_layer = tf.losses.softmax_cross_entropy(target_class, self.hidden_layer_weights)
        #loss_convolution = {
        #    3 : tf.losses.sigmoid_cross_entropy(un_truc_louche, self.kernels[3], weights = blame),
        #    4 : tf.losses.sigmoid_cross_entropy(un_truc_louche, self.kernels[4], weights = blame),
        #    5 : tf.losses.sigmoid_cross_entropy(un_truc_louche, self.kernels[5], weights = blame)
        #}

        self.optimize_step = self.optimiser.minimize(self.loss) #, var_list=[self.hidden_layer_weights]
        #[
            #, var_list=[self.hidden_layer_weights, self.kernels[3], self.kernels[4], self.kernels[5]]),#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
            #, self.kernels[3], self.kernels[4], self.kernels[5]
            #, var_list=[self.hidden_layer_weights, self.kernels[3], self.kernels[4], self.kernels[5]]
            #self.optimiser.minimize(loss_convolution_truc, var_list=[self.kernels[3]]
            #self.optimiser.minimize(self.loss_convolution[3], var_list=[self.convoluted_input_3]),#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
            #self.optimiser.minimize(self.loss_convolution[4], var_list=[self.convoluted_input_4]),#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
            #self.optimiser.minimize(self.loss_convolution[5], var_list=[self.convoluted_input_5])#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
        #]

        #Accuracy test
        self.predictions = tf.argmax(self.output, 1)
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.training_labels, 1))
        self.accuracy_value = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer(), {self.input_dictionnary : dictionnary})
        return

    def set_training_set(self, training_set, target_labels, batch_size):

        ### RUN SESSION ###
        print("Starting optimisation...")
        begin = 0
        end = batch_size
        while begin < len(training_set):

            self.session.run(self.optimize_step, feed_dict = {
                self.training_labels : target_labels[begin:end],
                self.input_sentence : training_set[begin:end]
            })
            #print(self.session.run(self.kernels[3])[0][0])
            begin += batch_size
            end += batch_size

        #print(session.run(self.hidden_layer_weights))
        #print(session.run(self.kernels[3][0]))
        print("Finished optimisation !")
        return

    def run(self, input_data, subdivision):

#        processed_input = self.__preprocess_input(input_data)
#
#        convoluted_input_3_values = processed_input[3]
#        convoluted_input_4_values = processed_input[4]
#        convoluted_input_5_values = processed_input[5]

        ### RUN SESSION ###
#        begin = 0
#        end = subdivision
#        while begin < len(input_data):
#            results.append(self.session.run(self.output, feed_dict =
#            {
#                self.input_sentence : input_data[begin:end]
#            }))
#            begin += subdivision
#            end += subdivision
#        
#        print(sum(results) / len(results))
        return results # DEBUG
    
    def accuracy(self, input, target, subdivision = 50):
        print("Success rate :")
        results = []
        begin = 0
        end = subdivision
        while begin < len(input):
            results.append(self.session.run(self.accuracy_value, feed_dict = {
                self.input_sentence : input[begin:end],
                self.training_labels : target[begin:end]
            }))
            begin += subdivision
            end += subdivision
        print(sum(results) / len(results))


#        trials = 0
#        success = 0
#        for i in xrange(len(input) - 1):
#            result = self.session.run(self.output, feed_dict = {
#                self.input_sentence : [input[i]]
#            })
##            print("{0} / {1}".format(result[0][0], target[i]))
#            if (result[0][0][0] > 0.5 and target[i][0] > 0.5) or (result[0][0][1] > 0.5 and target[i][1] > 0.5):
#                success += 1
#            trials += 1

        
        #print("{0} / {1} : {2}".format(success, trials, float(success) / float(trials)))
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
        return tf.truncated_normal(shape, stddev = 0.1)
