import tensorflow as tf
import numpy as np
from NeuralNetworkFunctions import initialize_neural_network


class ProofOfConceptNeuralNetwork:
    def _init_ (self, x, y, z, exact, X_f, layers, lower_bound, upper_bound):
        self.x = x
        self.y = y
        self.z = z

        self.exact = exact

        self.X_f = X_f

        self.layers = layers

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

                
        self.weights, self.biases = initialize_neural_network (layers)
