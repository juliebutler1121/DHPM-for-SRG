#########################################################################################
# SRG_1_Hidden_Layer.py
# Julie Butler
# April 2, 2019
# Verision 2.00
#
# Uses a deep neural network to predict the values of the SRG flow matrix at a given
# flow paramter.  The neural network is trained using data generated from the ode solver
# odeint.  
#
# TO-DO
# 1. Update comments
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
# For SRG and matrix manipulation
from numpy import array, dot, diag, reshape
from scipy.linalg import eigvalsh
from scipy.integrate import odeint
# Creating random numbers (for file naming)
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
# HAMILTONIAN
def Hamiltonian(delta,g):
    """
            Inputs:
                delta (a float): The energy level spacing for the system
                g (a float): The interaction strength for the system
            Returns:
                H (a 2D-array): The initial Hamiltonian for the system

            Creates and returns the initial Hamiltonian for a system given the energy
            level spacing and the interaction strength
    """

    H = array(
      [[2*delta-g,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g,          0.],
       [   -0.5*g, 4*delta-g,     -0.5*g,     -0.5*g,        0.,     -0.5*g ], 
       [   -0.5*g,    -0.5*g,  6*delta-g,         0.,    -0.5*g,     -0.5*g ], 
       [   -0.5*g,    -0.5*g,         0.,  6*delta-g,    -0.5*g,     -0.5*g ], 
       [   -0.5*g,        0.,     -0.5*g,     -0.5*g, 8*delta-g,     -0.5*g ], 
       [       0.,    -0.5*g,     -0.5*g,     -0.5*g,    -0.5*g, 10*delta-g ]]
    )
 
    return H

# COMMUTATOR
def commutator(a,b):
    """
        Inputs:
            a, b (2D-arrays): the matrices to take the commutator of (order dependent)
        Returns:
            Unnamed: the commutator of a and b
        Returns the commutator of matrices a and b
        [a,b] = ab - ba
        The order of the arguments is important
    """
    return dot(a,b) - dot(b,a)

# SRG_FLOW_EQUATION
def srg_flow_equation(y, s, dim):
    """
        Inputs:
            y (an array):  the current value of the Hamiltonian
            s (an array): the flow parameters values at which the SRG flow equation is
                to be solved at
            dim (an int): the dimension of one side of the square SRG matrix
        Returns:
            Unnamed (): the results from solving the SRG flow equation
        Solves the SRG flow equation at given values of the flow parameter.
        Taken from the code srg_pairing.py by H. Hergert
    """
 
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
def function_to_approximate (flow_parameters, SRG_matrices, times):
    """
        Inputs:
            flow_parameters (a 2D-array): the flow parameters at which SRG matrices are
                to be calculated.  Each inner array is simply a number, a single flow
                parameter.
            SRG_matrices (a 2D-array): SRG matrices for the 
                given flow parameters. Each inner array is 36 numbers long; they are
                flattened SRG matrices.
            times (an int): the number of times the training data is to be fed through 
                the neural network in a given iteration

        Returns:
            SRG_matrices or return_SRG_matrices (a 2D-array): SRG matrices for the 
                given flow parameters. Each inner array is 36 numbers long; they are
                flattened SRG matrices. If times = 1 then it is just the SRG matrices.
                If times is greater than 1 then it is the SRG matrices concatentated the
                given number of times.
        Returns the SRG matrices to train the neural network for the given flow parameter
        values.  The order and length of the flow parameters array must match the order
        and length of the SRG_matrices array.
    """        
    if times == 1:
        return SRG_matrices
    else:
        return_SRG_matrices = SRG_matrices
        for i in range(times-1):
            return_SRG_matrices = return_SRG_matrices + SRG_matrices

        return return_SRG_matrices

    
def main (hidden_dim, iterations, times, num_hidden_layers, ds_train, ds_predict):
    """
        Inputs:
            hidden_dim (an int): The number of neurons in the hidden layer
            iterations (an int): The number of training iteratons for the neural 
                network
            times (an int): The number of times the training data will be fed through
                the neural network in a given iteration
            num_hidden_layers (an int): The number of hidden layers
            ds_train (a float): The flow parameter step for the training data
            ds_predict (a float): The flow parameter step for the predictions
        Returns:
            None.
        Trains a neural network to approximate the SRG flow equation.  The neural network
        is trained with data produced from the ODE solver odeint.  The neural network   
        can have any number of hidden layers and any number of neurons per hidden layer
        but the number of neurons per hidden layer must be the same for each hidden layer.
    """
    # Pairing model set-up code taken from srg_pairing.py by H. Hergert

    # The intial Hamiltonian
    H0    = Hamiltonian(0.5, 1)
    dim   = H0.shape[0]

    # calculate exact eigenvalues
    eigenvalues = eigvalsh(H0)

    # turn initial Hamiltonian into a linear array
    y0  = reshape(H0, -1)                 

    # Create the training data

    # flow parameters for snapshot images
    flowparam_values_train = np.arange (0, 10, ds_train)

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys_train  = odeint(srg_flow_equation, y0, flowparam_values_train, args=(dim,))

    # Create the data to compare to the prediction values

    # flow parameters for snapshot images
    flowparam_values_predict = np.arange (0, 10, ds_predict)

    # integrate flow equations - odeint returns an array of solutions,
    # which are 1d arrays themselves
    ys_predict  = odeint(srg_flow_equation, y0, flowparam_values_predict, args=(dim,))

    # Neural Network

    # Create the Tensorflow computational graph
    with tf.variable_scope ('Graph'):
        # Placeholder for the values of s at which SRG matrices will be calculated
        # Given values when the Tensorflow session runs
        flow_params = tf.placeholder (tf.float32, shape=[None, 1], name='s_values')

        # Placeholder for the SRG matrices
        # Given values when the Tensorflow session runs
        SRG_matrices = tf.placeholder (tf.float32, shape=[None, 36], name='SRG_matrices')

        # The SRG matrices produced from the odeint solver
        y_true = function_to_approximate (flow_params, SRG_matrices, times)
        # The values of the SRG matrices approximated by the neural network
        y_approximate = ua (flow_params, 1, hidden_dim, 36, num_hidden_layers)

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
        results_folder = dir + '/results/srgapprox_' + str(int(time.time()))
        sw = tf.summary.FileWriter (results_folder, sess.graph)
        
        # Training the neural network
        print ('Training Universal Approximator:')
        # Start the Tensorflow Session
        sess.run (tf.global_variables_initializer ())
        # Train the neural network using 3000 iterations of training data
        for i in range (iterations):
            # The actual values that will be put into the placeholder input_vector
            flow_params_values = flowparam_values_train.reshape (len(flowparam_values_train), 
                1) 
            SRG_matrices_values = ys_train
            # Runs the Tensorflow session
            current_loss, loss_summary, _ = sess.run ([loss, loss_summary_t, 
                train_optimizer], feed_dict = { flow_params: flow_params_values,
                 SRG_matrices: SRG_matrices_values})

            sw.add_summary (loss_summary, i+1)

            # Print periodic updates to the terminal
            if (i+1)%250 == 0:
                print ('iteration: %d, loss: %f' % (i+1, current_loss))

        # Using the neural network to predict values of the SRG flow equation

        # The values of s to calculate matrices at
        prediction_values = np.arange (0, 10, ds_predict)
        prediction_values = prediction_values.reshape (len(prediction_values), 1)

        # Dummy variable for prediction algorithm        
        dummy_output = [[0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0,
                         0,0,0,0,0,0]]
     
        # Use the trained neural network to make the predictions
        y_true_results, y_approximate_results = sess.run ([y_true, y_approximate], 
            feed_dict={input_vector:prediction_values, output_vector:dummy_output})

        # Reshape for graphing
        prediction_values = prediction_values.flatten()

        # Graphing
        
        rc ('axes', linewidth=2)

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

        plot_title = 'Hidden Dim: ' + str(hidden_dim) + ', Iterations: ' + \
            str(iterations) + ', Times: ' + str(times) + ', Hidden Layers: ' + \
            str(num_hidden_layers)

        title (plot_title)

        # Calculated the difference between the odeint results and the prediction results
        # for each generated SRG matrix
        mag_differences = []
        for i in range (len (ys1)):
            mag_differences.append (np.linalg.norm (ys_predict[i]) - 
                np.linalg.norm(y_approximate_results[i])/times)

        # Find the average absolute difference between the odeint results and the 
        # prediction results
        print ("AVERAGE DIFFERENCE: ", np.mean (np.abs(mag_ys)))

        # Plot the differences against the flow paramter vlaues
        plot (prediction_values, mag_differences, 'b--', linewidth=4)      

       
        # Save the generated plot
        save_title = 'Graphs/SRG/' + str(hidden_dim) + '_' + str(times) + \
            '_' + str(iterations) + '_' + str(num_hidden_layers) + '_' + \
            str(random.randint(0, 100)) + '.png'

        savefig (save_title)

        #show()
        
#    saver.save (sess, results_folder + '/data.chkp')
           

# Runs when the program is called
if __name__=='__main__':
    main (hidden_dim=1000, iterations=3000, times=10, num_hidden_layers=5, 
        ds_train=0.1, ds_predict=0.01)



