import tensorflow as tf
import numpy as np

def initialize_neural_network (layers):
    weights = []
    biases = []

    number_of_layers = len(layers)

    for l in range (0, number_of_layers-1):
        w = xavier_initilization([layers[l], layers[l+1]])
        b = tf.Variable (tf.
