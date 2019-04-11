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
from pylab import *
from numpy import array, dot, diag, reshape
from scipy.linalg import eigvalsh
from scipy.integrate import odeint
import random

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

# FUNCTION_TO_APPROXIMATE
def function_to_approximate (input_vector, output_vector, times):
    if times == 1:
        return output_vector
    else:
        return_output_vector = output_vector
        for i in range(times-1):
            return_output_vector = return_output_vector + output_vector

        return return_output_vector

    
def main (hidden_dim, iterations, times, num_hidden_layers):
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

    # flow parameters for snapshot images
    flowparam_values1 = np.arange (0, 10, 0.01)
    flowparams1 = flowparam_values1

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys1  = odeint(derivative, y0, flowparams1, args=(dim,))




      # Create the Tensorflow computational graph
    with tf.variable_scope ('Graph'):
        # Placeholder for the values of x at which the value of sine will be calculated 
        # at
        # Given values when the Tensorflow session runs
        input_vector = tf.placeholder (tf.float32, shape=[None, 1], name='input_values')

        output_vector = tf.placeholder (tf.float32, shape=[None, 36], name='reference_values')

        # The actual values of sine at the selected values of x
        y_true = function_to_approximate (input_vector, output_vector, times)
        # The values of sine approximated by the neural network at the selected values
        # of x
        y_approximate = ua (input_vector, 1, hidden_dim, 36, num_hidden_layers)

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
            input_vector_values = flowparams.reshape (len(flowparams), 1) + flowparams.reshape (len(flowparams), 1)
            output_vector_values = ys
            # Runs the Tensorflow session
            current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                train_optimizer], feed_dict = {input_vector: input_vector_values, output_vector: output_vector_values})

            sw.add_summary (loss_summary, i+1)

            # Print periodic updates to the terminal
            if (i+1)%250 == 0:
                print ('iteration: %d, loss: %f' % (i+1, current_loss))

        # Using the neural network to predict values of sin(x)
        # The values of x to calculate sine at
        prediction_values = np.arange (0, 10, 0.01)
        dummy_output = [[0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0]]


        #print (dummy_output)

        prediction_values = prediction_values.reshape (len(prediction_values), 1)



        # Use the trained neural network to make the predictions
        y_true_results, y_approximate_results = sess.run ([y_true, y_approximate], feed_dict={input_vector:prediction_values, output_vector:dummy_output})

        prediction_values = prediction_values.flatten()

 
        #print (len (prediction_values), len (y_approximate_results))
        
        rc ('axes', linewidth=2)

        mag_y_approx = []

        for i in range (len (y_approximate_results)):
            #print (i)
            mag_y_approx.append (np.linalg.norm (y_approximate_results[i])/times)

        #print (len (prediction_values), len (mag_y_approx))     

        mag_ys = []

        for i in range (len (ys1)):
            mag_ys.append (np.linalg.norm (ys1[i]) - mag_y_approx[i])

        print ("AVERAGE DIFFERENCE: ", np.mean (np.abs(mag_ys)))

           
        plot (prediction_values, mag_ys, 'b--', linewidth=4, label = 'NN')
        #plot (flowparams1, mag_ys, 'g', linewidth=4, label='SRG')
        

        fontsize = 12

        ax = gca ()

        for tick in ax.xaxis.get_major_ticks():
        	tick.label1.set_fontsize (fontsize)
        	tick.label1.set_fontweight ('bold')
        for tick in ax.yaxis.get_major_ticks ():
        	tick.label1.set_fontsize (fontsize)
        	tick.label1.set_fontweight ('bold')

        xlabel ('Flow Parameter', fontsize=16, fontweight='bold')
        ylabel ('Difference from ODE Result', fontsize=16, fontweight='bold')

        plot_title = 'Hidden Dim: ' + str(hidden_dim) + ', Iterations: ' + str(iterations) + ', Times: ' + str(times) + ', Hidden Layers: ' + str(num_hidden_layers)

        title (plot_title)

        save_title = 'Graphs/SRG' + str(hidden_dim) + '_' + str(times) +  '_' + str(iterations) + '_' + str(num_hidden_layers) + str(random.randint(0, 100)) + '.png'

        savefig (save_title)

        legend (fontsize=14)

        #show()
        
#    saver.save (sess, results_folder + '/data.chkp')
           

# Runs when the program is called
if __name__=='__main__':
    main (1000, 3000, 10, 5)



