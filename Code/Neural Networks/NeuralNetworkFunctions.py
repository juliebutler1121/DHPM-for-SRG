import tensorflow as tf
import numpy as np

def initialize_neural_network (layers):
    weights = []
    biases = []

    number_of_layers = len(layers)

    for l in range (0, number_of_layers-1):
        w = xavier_initilization(layers[l], layers[l+1])
        b = tf.Variable (tf.zeros([1, layers[l+1]]))

        weights.append (w)
        biases.append (b)

    return weights, biases

def xavier_initilization (layer_in, layer_out):
    xavier_standard_deviation = np.sqrt(2/(layer_in + layer_out))

    return tf.Variable (tf.truncated_normal([layer_in, layer_out], 
        stddev=xavier_standard_deviation))
    
# UNIVERSAL_FUNCTION_APPROXIMATOR
def universal_function_approximator_one_hidden_layer (input_vector, input_dim, hidden_dim, output_dim):
    """
        Approximates any function using a one hidden layer neural network.
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
            
        z = tf.matmul (activated_function, weights_output_layer)
        
        return z
