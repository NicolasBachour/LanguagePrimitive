
# ADD THE BIAS WEIGHT to each kernel
# ADD THE ACTIVATION FUNCTION

import numpy as np
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, vector_dimention, epochs):
        ### Network settings ###
        self.vector_dimention = vector_dimention
        self.epochs = epochs # DEBUG : unused ?

        ### Creating weights ###
        # Create 100 kernels of 3 * k, 100 kernels of 4 * k, 100 kernels of 5 * k
        self.kernels = {
            3 : tf.Variable(self.__shape_to_variables([3 * self.vector_dimention, 100]), tf.float32),
            4 : tf.Variable(self.__shape_to_variables([4 * self.vector_dimention, 100]), tf.float32),
            5 : tf.Variable(self.__shape_to_variables([5 * self.vector_dimention, 100]), tf.float32)
        }

        # Create 300 * 2 weights for the output layer (2 classes : positive + negative)
        self.hidden_layer_weights = tf.Variable(self.__shape_to_variables([300, 2]), tf.float32)

        ### Build network graph ###
        self.convoluted_input_3 = tf.placeholder(tf.float32, shape = [None, None, 1, 3 * self.vector_dimention]) # of shape batches, (n - h + 1), 1, hk
        self.convoluted_input_4 = tf.placeholder(tf.float32, shape = [None, None, 1, 4 * self.vector_dimention]) # of shape batches, (n - h + 1), 1, hk
        self.convoluted_input_5 = tf.placeholder(tf.float32, shape = [None, None, 1, 5 * self.vector_dimention]) # of shape batches, (n - h + 1), 1, hk

        # Produce feature maps (of shape [k - s + 1, 100, 300])
        convolution_3_feature_map = tf.tanh(tf.matmul(self.convoluted_input_3, tf.tile(tf.expand_dims(tf.expand_dims(self.kernels[3], 0), 0), [tf.shape(self.convoluted_input_3)[0], tf.shape(self.convoluted_input_3)[1], 1, 1]))) # of shape [batches, n - h + 1, 1, 100]
        convolution_4_feature_map = tf.tanh(tf.matmul(self.convoluted_input_4, tf.tile(tf.expand_dims(tf.expand_dims(self.kernels[4], 0), 0), [tf.shape(self.convoluted_input_4)[0], tf.shape(self.convoluted_input_4)[1], 1, 1]))) # of shape [batches, n - h + 1, 1, 100]
        convolution_5_feature_map = tf.tanh(tf.matmul(self.convoluted_input_5, tf.tile(tf.expand_dims(tf.expand_dims(self.kernels[5], 0), 0), [tf.shape(self.convoluted_input_5)[0], tf.shape(self.convoluted_input_5)[1], 1, 1]))) # of shape [batches, n - h + 1, 1, 100]

        # Max overtime pooling
        feature_3 = tf.reduce_max(convolution_3_feature_map, reduction_indices = [1]) # of size [batches, 1, 100]
        feature_4 = tf.reduce_max(convolution_4_feature_map, reduction_indices = [1]) # of size [batches, 1, 100]
        feature_5 = tf.reduce_max(convolution_5_feature_map, reduction_indices = [1]) # of size [batches, 1, 100]
        # DEBUG : Sees tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')    for alternative

        # Output
        # Concatenate along axis 2
        feature_vector = tf.concat(2, [feature_3, feature_4, feature_5]) # of size [batches, 1, 300]
        output_without_softmax = tf.matmul(feature_vector, tf.tile(tf.expand_dims(self.hidden_layer_weights, 0), [tf.shape(self.convoluted_input_3)[0], 1, 1])) # of size [batches, 2, 1]
        self.output = tf.nn.softmax(output_without_softmax)
        return

    def set_training_set(self, training_set, target_labels, batch_size):
        # training_set is a table containing inputs
        # target_labels is a table containing target outputs

        number_of_classes = len(target_labels[0]) # number of classes
        # batch_size : number of trials
        # target_labels : vector of length N with values corresponding to the correct class

        # Hand down the weights update to Adadelta
        #indicators = np.zeros((batch_size, number_of_classes))
        #indicators[np.arange(batch_size), target_labels] = 1

        #output_matrix = []
        #for i in xrange(0, batch_size):
        #    output_matrix.append(self.output())

        self.loss_hidden_layer = - tf.reduce_sum(tf.multiply(target_labels, tf.log(self.output)))

        #loss_hidden_layer = tf.losses.softmax_cross_entropy(target_class, self.hidden_layer_weights)
        #loss_convolution = {
        #    3 : tf.losses.sigmoid_cross_entropy(un_truc_louche, self.kernels[3], weights = blame),
        #    4 : tf.losses.sigmoid_cross_entropy(un_truc_louche, self.kernels[4], weights = blame),
        #    5 : tf.losses.sigmoid_cross_entropy(un_truc_louche, self.kernels[5], weights = blame)
        #}

        self.optimiser = tf.train.AdadeltaOptimizer(learning_rate = 1.0)
        self.optimize_step = [
            self.optimiser.minimize(self.loss_hidden_layer, var_list=[self.hidden_layer_weights]),#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
            #self.optimiser.minimize(self.loss_convolution[3], var_list=[self.convoluted_input_3]),#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
            #self.optimiser.minimize(self.loss_convolution[4], var_list=[self.convoluted_input_4]),#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
            #self.optimiser.minimize(self.loss_convolution[5], var_list=[self.convoluted_input_5])#, aggregation_method=tf.AggregationMethod.ADD_N, reduction=tf.losses.Reduction.SUM)
        ]

        convoluted_input_3_values = []
        convoluted_input_4_values = []
        convoluted_input_5_values = []
        for sample in training_set:
            processed_input = self.__preprocess_input(sample)
            convoluted_input_3_values.append(processed_input[3])
            convoluted_input_4_values.append(processed_input[4])
            convoluted_input_5_values.append(processed_input[5])

        self.global_initialiser = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(self.global_initialiser) # DEBUG ?
            results = session.run(self.optimize_step, feed_dict = {
                self.convoluted_input_3 : convoluted_input_3_values,
                self.convoluted_input_4 : convoluted_input_4_values,
                self.convoluted_input_5 : convoluted_input_5_values
            })
            print(results)

        #fc1 = tf.nn.dropout(fc1, dropout)
        return

    def run(self, input):
        results = self.__run(input)
        return results # DEBUG
        return results.output

    def __preprocess_input(self, input_data):
        sentence_length = len(input_data)

        # If input data is too short, add padding
        while sentence_length < 5:
            input_data.append(np.zeros(self.vector_dimention, dtype=np.float32))
            sentence_length += 1

        # Tile the input matrix in the three "convoluted_input"
        return {
            3 : self.__map_input(input_data, 3),
            4 : self.__map_input(input_data, 4),
            5 : self.__map_input(input_data, 5)
        }

    def __run(self, input_data):

        processed_input = self.__preprocess_input(input_data)

        convoluted_input_3_values = processed_input[3]
        convoluted_input_4_values = processed_input[4]
        convoluted_input_5_values = processed_input[5]

        with tf.Session() as session:
            session.run(self.global_initialiser)
                                                                                    #results = session.run(output_without_softmax, feed_dict=
                                                                                    #{
                                                                                    #    convoluted_input_3 : convoluted_input_3_values,
                                                                                    #    convoluted_input_4 : convoluted_input_4_values,
                                                                                    #    convoluted_input_5 : convoluted_input_5_values
                                                                                    #}) #DEBUG
                                                                                    #print(results)
            results = session.run(self.output, feed_dict=
            {
                self.convoluted_input_3 : [convoluted_input_3_values],
                self.convoluted_input_4 : [convoluted_input_4_values],
                self.convoluted_input_5 : [convoluted_input_5_values]
            })
            print(results) # DEBUG
        return results # DEBUG

    def train(self, input_data, target_class):
        # Train the neural network given the sample

        for i in xrange(0, len(input_data)):
            input_data_p.append(self.__preprocess_input(input_data))

#        for i in xrange(0, self.epochs):
#            self.optimize_step

        # Launch the graph
        with tf.Session() as session:
            session.run(init)
            step = 1
            # Keep training until reach max iterations
            while step * 50 < training_iters: # DEBUG : batch size of 50
                #batch_x, batch_y = mnist.train.next_batch(batch_size)

                # Run optimization op (backprop)
                sess.run(self.optimize_step, feed_dict = {input_data_p[0], target_class[0]})
                #if step % display_step == 0:
                #    # Calculate batch loss and accuracy
                #    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                #                                                      y: batch_y,
                #                                                      keep_prob: 1.})
                #    print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                #          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                #          "{:.5f}".format(acc))
                step += 1
            print("Optimization Finished!")
        
            # Calculate accuracy for 256 mnist test images
#            print("Testing Accuracy:", \
#                sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
#                                              y: mnist.test.labels[:256],
#                                              keep_prob: 1.}))

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


