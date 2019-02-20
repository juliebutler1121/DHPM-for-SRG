import tensorflow as tf
import numpy as np
import matplotlib.pyploy as plt

def universal_function_approximator (input_vector, input_dim, hidden_dim, output_dim):
    with tf.variable_scope ('UniversalApproximator'):
        weights_hidden_layer = tf.get_variable (name='weights_hidden_layer', shape=[input_dim, hidden_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
        
        biases_hidden_layer = tf.get_variable (name='biases_hidden_layer', shape=[hidden_dim],
            initializer=tf.constant_initializer(0.0))
        
        z = tf.matmul (input_vector, weights_hidden_layer) + biases_hidden_layer
        activated_function = tf.nn.relu (z)
        
        weights_output_layer = tf.get_variable (name='weights_output_layer', shape=[hidden_dim, output_dim],
            initializer=tf.random_normal_initializer(stddev=0.1))
            
        z = tf.matmul (activated_function, weights_output_layer)
        
        return z
        
def function_to_approximate (x):  
    return tf.sin (x)
    
