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
import matplotlib.pyploy as plt

#LOCAL IMPORTS
from NeuralNetworkFunctions import universal_function_approximator_one_hidden_layer as ua

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
    
