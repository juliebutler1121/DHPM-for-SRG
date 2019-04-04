########################################################################################
# NeuralNetworkFunctions.py
# Julie Butler
# February 20, 2019
# Version 0.2
#
# A collection of functions for setting up and running neural networks.  Many are 
# related to the use of neural networks as universal function approximators.

# FUNCTIONS:
# initilize_neural_network:  Creates the weights and biases of a neural network, for any
#   number of hidden layers given in an inputted array of seed values for the weights and
#   biases.
# xavier_initialization: initializes the weights of a neural network using the Xavier
#   initialization method
# universal_function_approximator: approximates a function of any input and output 
#   dimensions using a one hidden layer neural network with a specified number of 
#   neurons.
########################################################################################

#############################
#                           #
# IMPORTS                   #
#                           #
#############################
# THIRD-PARTY IMPORTS
# For machine learning
import tensorflow as tf
# For matrix calculations
import numpy as np

#INITIALIZE_NEURAL_NETWORK
def initialize_neural_network (layers):
    """
        Inputs:
            layers (an array of ints): Seed values for the Xavier Initialization.  There
                should be one number for each layer in the neural network, including the
                input layer
        Returns:
            weights (an array): The weights for each layer in the neural network
            biases (an array): The biases for each layer in the neural network
        Calculated the weights and biases for a neural network with one or more hidden 
        layers using the Xavier initialization method.  
    """
    weights = []
    biases = []

    number_of_layers = len(layers)

    # Calculates weights and biases for each hidden layer and the outout.  len(weights) 
    # is one less than len(layers) (no weights or biases for the input layer)
    for l in range (0, number_of_layers-1):
        w = xavier_initialization(layers[l], layers[l+1])
        b = tf.Variable (tf.zeros([1, layers[l+1]]))

        weights.append (w)
        biases.append (b)

    return weights, biases

#XAVIER_INITIALIZATION
def xavier_initialization (layer_in, layer_out):
    """
        Inputs:
            layer_in (an int): seed value for the Xavier initialization
            layer_out (an int): seed value for the Xavier initialization
        Returns: 
            Unnamed (Tensorflow variable): the weight for the layer
        Calculates the weights for a layer in a neural network using the Xavier
        initialization method.
    """
    xavier_standard_deviation = np.sqrt(2/(layer_in + layer_out))

    return tf.Variable (tf.truncated_normal([layer_in, layer_out], 
        stddev=xavier_standard_deviation))
    
# UNIVERSAL_FUNCTION_APPROXIMATOR_ONE_HIDDEN_LAYER
def universal_function_approximator_one_hidden_layer (input_vector, input_dim, hidden_dim, output_dim):
    """
        Approximates any function using a one hidden layer neural network. Weights and 
        biases are initialized using a Tensorflow initializer.  
        Inputs:
            input_vector (an array or matrix): the input values at which the function is
                to be calculated.  Can be a one dimensional array if approximating a one
                dimensional function (f(x)) or can be a matrix if approximating a 
                multi-dimensional array (f(x, y, z)).
            input_dim (an int): the length of the inner arrays of input_vector (1 if
                input_vector is an array).  The number of input variables needed by
                the function to approximate.  Ex: input_dim is 1 to approximate f(x)
                and is 3 to approximate f(x, y, z).
            hidden_dim (an int): the number of neurons in the hiddeb layer.  In general
                a higher value of hidden_dim will results in a better approximation.
            output_dim (an int): the number of variables outputted by approximating the 
                function.  Generally will be one.
    """
    # Makes sure variables are not redefined
    with tf.variable_scope ('UniversalApproximator'):
        weights_hidden_layer = tf.get_variable (name='weights_hidden_layer', 
            shape=[input_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer = tf.get_variable (name='biases_hidden_layer', 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (input_vector, weights_hidden_layer) + biases_hidden_layer
        activated_function = tf.nn.relu (z)
        
        weights_output_layer = tf.get_variable (name='weights_output_layer', 
            shape=[hidden_dim, output_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
            
        # See notebook #1 page 22
        z = tf.matmul (activated_function, weights_output_layer)
        
        return z

def universal_function_approximator_N_hidden_layers (input_vector, input_dim, hidden_dim, output_dim, num_hidden_layers):
    # First Hidden Layer
    weights_first_hidden_layer = tf.get_variable (name='weights_first_hidden_layer', 
        shape=[input_dim, hidden_dim],
        initializer=tf.random_normal_initializer(stddev=0.1))
        
    biases_first_hidden_layer = tf.get_variable (name='first_biases_hidden_layer', 
        shape=[hidden_dim],
        initializer=tf.constant_initializer(0.0))
        
    z = tf.matmul (input_vector, weights_first_hidden_layer) + biases_first_hidden_layer
    activated_function = tf.nn.relu (z)

    for i in range (0, num_hidden_layers-1):
        # Interior Hidden Layers
        weight_name = 'weight_hidden_layer_' + str(i)
        bias_name = 'bias_hidden_layer_' + str(i)

        weights_hidden_layer_1 = tf.get_variable (name=weight_name, 
            shape=[hidden_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer_1 = tf.get_variable (name=bias_name, 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (activated_function, weights_hidden_layer_1) + biases_hidden_layer_1
        activated_function = tf.nn.relu (z)

    #Output Layer
    weights_output_layer = tf.get_variable (name='weights_output_layer', 
        shape=[hidden_dim, output_dim],
        initializer=tf.random_normal_initializer(stddev=0.1))
            
    # See notebook #1 page 22
    z = tf.matmul (activated_function, weights_output_layer)
        
    return z
"""
PANIC VERSION -- WORKS WITH 2 HIDDEN LAYERS
# UNIVERSAL_FUNCTION_APPROXIMATOR_N_HIDDEN_LAYERS
def universal_function_approximator_N_hidden_layers (input_vector, input_dim, hidden_dim, output_dim, num_hidden_layers, lower_bound, upper_bound):
        # First Hidden Layer
        weights_hidden_layer = tf.get_variable (name='weights_hidden_layer', 
            shape=[input_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer = tf.get_variable (name='biases_hidden_layer', 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (input_vector, weights_hidden_layer) + biases_hidden_layer
        activated_function = tf.nn.relu (z)

        #Second Hidden Layer
        weights_hidden_layer_1 = tf.get_variable (name='weights_hidden_layer_1', 
            shape=[hidden_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer_1 = tf.get_variable (name='biases_hidden_layer_1', 
            shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (activated_function, weights_hidden_layer_1) + biases_hidden_layer_1
        activated_function = tf.nn.relu (z)

        #Output Layer
        weights_output_layer = tf.get_variable (name='weights_output_layer', 
            shape=[hidden_dim, output_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
            
        # See notebook #1 page 22
        z = tf.matmul (activated_function, weights_output_layer)
        
        return z
"""    
    
