#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """
        -nx is the number of input features
        -layers is a list representing the number of
         nodes in each layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            b = np.zeros((layers[i], 1))
            if i == 0:
                weight = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                f1 = np.random.randn(layers[i], layers[i - 1])
                f2 = np.sqrt(2 / layers[i - 1])
                weight = f1 * f2
            weights["W" + str(i + 1)] = weight
            weights["b" + str(i + 1)] = b
        self.__weights = weights

    @property
    def L(self):
        """Returns number of layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """Returns a dictionary with all intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Returns a dictionary with all weights and biased of the network"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(self.__L):
            w = self.weights['W' + str(i + 1)]
            b = self.weights['b' + str(i + 1)]
            if i == 0:
                z = np.matmul(w, X) + b
            else:
                z = np.matmul(w, A) + b
            A = 1 / (1 + np.exp(-z))
            self.__cache['A' + str(i + 1)] = A
        return A, self.cache
