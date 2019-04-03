#########################################################################################
# SRG_1_Hidden_Layer.py
# Julie Butler
# April 2, 2019
# Verision 1.00
#
#
# To-Do:
# Add predicition function
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
from numpy import array, dot, diag, reshape
from scipy.linalg import eigvalsh
from scipy.integrate import odeint

# LOCAL IMPORTS
from NeuralNetworkFunctions import universal_function_approximator_one_hidden_layer as ua

# SYSTEM IMPORTS
import time, os
dir = os.path.dirname (os.path.realpath (__file__))

class Function():
    def __init__ (self, dictionary):
        self.dictionary = dictionary 

    def get_value (self, x):
        x = tf.to_float(x)
        print ('***************', x)
        return self.dictionary[x]

#############################
#                           #
# FUNCTIONS                 #
#                           #
#############################
# Hamiltonian for the pairing model
def Hamiltonian(delta,g):

  H = array(
      [[2*delta-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,          0.],
       [   -0.5*g, 4*delta-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ], 
       [   -0.5*g,    -0.5*g,  6*delta-g,         0.,    -0.5*g,     -0.5*g ], 
       [   -0.5*g,    -0.5*g,         0.,  6*delta-g,    -0.5*g,     -0.5*g ], 
       [   -0.5*g,        0.,     -0.5*g,     -0.5*g, 8*delta-g,     -0.5*g ], 
       [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g, 10*delta-g ]]
    )

  return H

# commutator of matrices
def commutator(a,b):
  return dot(a,b) - dot(b,a)

# derivative / right-hand side of the flow equation
def derivative(y, t, dim):
 
  # reshape the solution vector into a dim x dim matrix
  H = reshape(y, (dim, dim))

  # extract diagonal Hamiltonian...
  Hd  = diag(diag(H))

  # ... and construct off-diagonal the Hamiltonian
  Hod = H-Hd

  # calculate the generator
  eta = commutator(Hd, Hod)

  # dH is the derivative in matrix form 
  dH  = commutator(eta, H)

  # convert dH into a linear array for the ODE solver
  dydt = reshape(dH, -1)
    
  return dydt

    
def main (hidden_dim):
    """
        Inputs:
            hidden_dim (an int): The number of neurons in the hidden layer
        Returns:
            None.
        Trains a neural network of one hidden layer to approxiamate sin(x)
    """
    H0    = Hamiltonian(0.5, 1)
    dim   = H0.shape[0]

    # calculate exact eigenvalues
    eigenvalues = eigvalsh(H0)

    # turn initial Hamiltonian into a linear array
    y0  = reshape(H0, -1)                 

    # flow parameters for snapshot images
    flowparam_values = np.arange (0, 10, 0.1)
    flowparams = flowparam_values

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys  = odeint(derivative, y0, flowparams, args=(dim,))

    for s in flowparams:
        s - tf.convert_to_tensor (s)

    dictionary = dict (zip (flowparams, ys))

    function = Function(dictionary)

    print (type(function.get_value (4)[0]))

    

    # Create the Tensorflow computational graph
    with tf.variable_scope ('Graph'):
        # Placeholder for the values of x at which the value of sine will be calculated 
        # at
        # Given values when the Tensorflow session runs
        input_vector = tf.placeholder (tf.float64, shape=[None, 1], name='input_values')

        # The actual values of sine at the selected values of x
        y_true = function.get_value (input_vector)
        # The values of sine approximated by the neural network at the selected values
        # of x
        y_approximate = ua (input_vector, 1, hidden_dim, 36)

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
        for i in range (3000):
            # The actual values that will be put into the placeholder input_vector
            input_vector_values = flowparam_values
            # Runs the Tensorflow session
            current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                train_optimizer], feed_dict = {input_vector: input_vector_values})

            sw.add_summary (loss_summary, i+1)

            # Print periodic updates to the terminal
            if (i+1)%100 == 0:
                print ('iteration: %d, loss: %f' % (i+1, current_loss))
#    saver.save (sess, results_folder + '/data.chkp')
           

# Runs when the program is called
if __name__=='__main__':
    main (100)
