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
    
