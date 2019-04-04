#########################################################################################
# SineApproximator.py
# Julie Butler
# February 20, 2019
# Verision 2.5
#
# Approximates the function sin(x) using a one hidden layer neural network with any
# number of neurons.  In general, as the number of neurons increases, the accuracy
# of the approximation increases
#
# To-Do:
# Adapt for multi-hidden-layer NN
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
from pylab import *
# 
from math import sin

# LOCAL IMPORTS
from NeuralNetworkFunctions import universal_function_approximator_N_hidden_layers as ua

# SYSTEM IMPORTS
import time, os
dir = os.path.dirname (os.path.realpath (__file__))
#############################
#                           #
# FUNCTIONS                 #
#                           #
#############################
# FUNCTION_TO_APPROXIMATE
def function_to_approximate (x):
    """
        Inputs:
            x (a Tensorflow Variable): the point to calculate the function at
        Returns:
            Unnamed (a Tensorflow Variable): the value of sine at the inputted point
        Calculates sin(x) for a given value of x.
    """  
    return tf.sin(x)
    
def main (hidden_dim, training_points, iterations, num_hidden_layers):
    """
        Inputs:
            hidden_dim (an int): The number of neurons in the hidden layer
            training_points (an int): The number of training points per iteration
            iterations (an int): The number of iterations during the training process.
        Returns:
            None.
        Trains a neural network of one hidden layer to approxiamate sin(x)
    """

    print ('*************** HIDDEN LAYERS: ', num_hidden_layers, '********************')
    # Create the Tensorflow computational graph
    with tf.variable_scope ('Graph'):
        # Placeholder for the values of x at which the value of sine will be calculated 
        # at
        # Given values when the Tensorflow session runs
        input_vector = tf.placeholder (tf.float32, shape=[None, 1], name='input_values')

        # The actual values of sine at the selected values of x
        y_true = function_to_approximate (input_vector)
        # The values of sine approximated by the neural network at the selected values
        # of x
        y_approximate = ua (input_vector, 1, hidden_dim, 1, num_hidden_layers)

        # Function used to train the neural network
        with tf.variable_scope ('Loss'):
            # Cost function        
            loss=tf.reduce_mean (tf.square (y_approximate-y_true))
            loss_summary_t = tf.summary.scalar ('loss', loss)

        # Optimizer, uses an Adam optimizer
        adam = tf.train.AdamOptimizer (learning_rate = 1e-2)
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
        for i in range (iterations):
            # The actual values that will be put into the placeholder input_vector
            input_vector_values = np.random.uniform (-10, 10, [training_points, 1])
            # Runs the Tensorflow session
            current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                train_optimizer], feed_dict = {input_vector: input_vector_values})

            sw.add_summary (loss_summary, i+1)

            # Print periodic updates to the terminal
            if (i+1)%100 == 0:
                print ('iteration: %d, loss: %f' % (i+1, current_loss))

        # Using the neural network to predict values of sin(x)
        # The values of x to calculate sine at
        prediction_values = np.arange (-10, 10, 0.1)

        prediction_values = prediction_values.reshape (len(prediction_values), 1)

        # Use the trained neural network to make the predictions
        y_true_results, y_approximate_results = sess.run ([y_true, y_approximate], feed_dict={input_vector:prediction_values})

        prediction_values = prediction_values.flatten()

        y_approximate_results = y_approximate_results.flatten()

        rc ('axes', linewidth=2)

        plot (prediction_values, y_approximate_results, 'b--', linewidth=4, label='NN Results')
       
        actual_results = np.sin (prediction_values)

        plot (prediction_values, actual_results, 'g', linewidth=4, label='sin(x)')

        fontsize = 12

        ax = gca ()

        for tick in ax.xaxis.get_major_ticks():
        	tick.label1.set_fontsize (fontsize)
        	tick.label1.set_fontweight ('bold')
        for tick in ax.yaxis.get_major_ticks ():
        	tick.label1.set_fontsize (fontsize)
        	tick.label1.set_fontweight ('bold')

        xlabel ('X', fontsize=16, fontweight='bold')
        ylabel ('Y', fontsize=16, fontweight='bold')

        plot_title = 'Iterations: ' + str(iterations) + ', Training Points: ' + str(training_points) + ', Hidden Layers: ' + str(num_hidden_layers)

        title (plot_title)

        save_title = 'Graphs/Multi_Hidden_Layers/' + str(hidden_dim) + '_' + str(training_points) + '_' + str(iterations) + '_' + str(num_hidden_layers)+'.png'

        savefig (save_title)

        legend (fontsize=14)

        print ('************************************************')

        #show()
        
    #saver.save (sess, results_folder + '/data.chkp')
            

# Runs when the program is called
if __name__=='__main__':
    #main (500, 10000, 3000, 1)
    main (500, 10000, 3000, 2)
    #main (500, 10000, 3000, 5)
    #main (500, 10000, 3000, 10)
