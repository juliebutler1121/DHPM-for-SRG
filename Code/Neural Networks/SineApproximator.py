#########################################################################################
# SineApproximator.py
# Julie Butler
# February 20, 2019
# Verision 0.2
#
# Approximates the function sin(x) using a one hidden layer neural network with any
# number of neurons.  In general, as the number of neurons increases, the accuracy
# of the approximation increases
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
from NeuralNetworkFunctions import universal_function_approximator_one_hidden_layer as ua

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
    return tf.sin (x)
    
def main (hidden_dim):
    # Create the Tensorflow computational graph
    print ('entered main')
    with tf.variable_scope ('Graph'):
        print ('Graph')
        input_vector = tf.placeholder (tf.float32, shape=[None, 1], name='input_values')
        y_true = function_to_approximate (input_vector)
        y_approximate = ua (input_vector, 1, hidden_dim, 1)
        with tf.variable_scope ('Loss'):
            loss=tf.reduce_mean (tf.square (y_approximate-y_true))
            loss_summary_t = tf.summary.scalar ('loss', loss)
        adam = tf.train.AdamOptimizer (learning_rate = 1e-2)
        train_optimizer = adam.minimize (loss)
    saver = tf.train.Saver()
    print ('Session')
    with tf.Session() as sess:
        results_folder = dir + '/results/sinapprox_' + str(int(time.time()))
        sw = tf.summary.FileWriter (results_folder, sess.graph)
        print ('Training Universal Approximator:')
        sess.run (tf.global_variables_initializer ())
        for i in range (3000):
            input_vector_values = np.random.uniform (-10, 10, [10000, 1])
            current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                train_optimizer], feed_dict = {input_vector: input_vector_values})
            sw.add_summary (loss_summary, i+1)
            if (i+1)%100 == 0:
                print ('batch: %d, loss: %f' % (i+1, current_loss))
#    saver.save (sess, results_folder + '/data.chkp')
            


if __name__=='__main__':
    main (200)
