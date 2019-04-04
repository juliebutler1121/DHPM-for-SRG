#########################################################################################
# UniversalApproximator.py
# Julie Butler
# February 28, 2019
# Verision 1.00
#
# Approximates the function sin(x) using a one hidden layer neural network with any
# number of neurons.  In general, as the number of neurons increases, the accuracy
# of the approximation increases
#
# To-Do:
# Add predicition function
# Have Functions function return input and output dimensions
#########################################################################################

#############################
#                           #
# IMPORTS                   #
#                           #
#############################
# THIRD-PARTY IMPORTS
# For machine learning
import tensorflow as tf
# For calculations
import numpy as np
# For graphing
import matplotlib.pyplot as plt

# LOCAL IMPORTS
# Neural network that functions as a function approximator
from NeuralNetworkFunctions import universal_function_approximator_one_hidden_layer as ua
# Function to approximate
from Functions import test_vector as function_to_approximate
from Functions import test_vector_dims as dimensions

# SYSTEM IMPORTS
import time, os
dir = os.path.dirname (os.path.realpath (__file__))
#############################
#                           #
# FUNCTIONS                 #
#                           #
#############################
def main (hidden_dim, num_iterations, training_points, 
    learning_rate):
    """
        Inputs:
            hidden_dim (an int): The number of neurons in the hidden layer
        Returns:
            None.
        Trains a neural network of one hidden layer to approxiamate sin(x)
    """
    input_dim, output_dim = dimensions()
    # Create the Tensorflow computational graph
    with tf.variable_scope ('Graph'):
        # Placeholder for the values of x at which the value of sine will be calculated 
        # at
        # Given values when the Tensorflow session runs
        input_vector = tf.placeholder (tf.float32, shape=[None, input_dim], name='input_values')

        # The actual values of sine at the selected values of x
        y_true = function_to_approximate (input_vector)
        # The values of sine approximated by the neural network at the selected values
        # of x
        y_approximate = ua (input_vector, input_dim, hidden_dim, output_dim)

        # Function used to train the neural network
        with tf.variable_scope ('Loss'):
            # Cost function
            loss=tf.reduce_mean (tf.square (y_approximate-y_true))
            loss_summary_t = tf.summary.scalar ('loss', loss)

        # Optimizer, uses an Adam optimizer
        adam = tf.train.AdamOptimizer (learning_rate = learning_rate)
        # Minimize the cost function using the Adam optimizer
        train_optimizer = adam.minimize (loss)

    # Saves a training session
    saver = tf.train.Saver()
    # Tensorflow Session (what acutally runs the neural network)
    with tf.Session() as sess:
        # Where to store the results of the neural network (currently not implemented)
        results_folder = dir + '/results/sinapprox_' + str(int(time.time()))
        sw = tf.summary.FileWriter (results_folder, sess.graph)
        
        # Training the neural network
        print ('Training Universal Approximator:')
        # Start the Tensorflow Session
        sess.run (tf.global_variables_initializer ())
        # Train the neural network using 3000 iterations of training data
        for i in range (num_iterations):
            # The actual values that will be put into the placeholder input_vector
            #input_vector_values = [0]           
            input_vector_values = np.random.uniform (-10, 10, [training_points, input_dim])
            # Runs the Tensorflow session
            current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                train_optimizer], feed_dict = {input_vector: input_vector_values})

            sw.add_summary (loss_summary, i+1)

            # Print periodic updates to the terminal
            if (i+1)%100 == 0:
                print ('batch: %d, loss: %f' % (i+1, current_loss))
#    saver.save (sess, results_folder + '/data.chkp')
            

# Runs when the program is called
if __name__=='__main__':
    main (500, 3000, 10000, 1e-2)
