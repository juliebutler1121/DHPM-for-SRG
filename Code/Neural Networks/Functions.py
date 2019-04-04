import numpy as np
import tensorflow as tf
def sine_1d (x):
    return tf.sin(x)
def sine_1d_dims ():
    return 1, 1

def sine_3d (coordinates):
    return tf.sin(coordinates[0])*tf.sin(coordinates[1])*tf.sin(coordinates[2])
def sine_3d_dims ():
    return 3, 1

def test_vector (x):
    test = [0, 1, 2, 3]
    print(type(tf.to_float(x)))
    return 4
def test_vector_dims ():
    return 1, 1

def quadratic (x):
    return x**2
def quadratic_dims ():
    return 1, 1

def test_matrix (x):
    return np.array([tf.sin(x), 0], [0, tf.cos(x)])
def test_matrix_dims ():
    return 1, 2
