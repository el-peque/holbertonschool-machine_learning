#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        """
        nx is the number of input features
        layers is a list representing the number of
        nodes in each layer of the network
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(i, int) and i > 0 for i in layers):
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        weights = {}
        for layer in range(len(layers)):
            b = np.zeros((layers[layer], 1))
            if layer == 0:
                weight = np.random.randn(layers[layer], nx) * np.sqrt(2 / nx)
            else:
                f1 = np.random.randn(layers[layer], layers[layer - 1])
                f2 = np.sqrt(2 / layers[layer - 1])
                weight = f1 * f2
            weights["b" + str(layer + 1)] = b
            weights["W" + str(layer + 1)] = weight
        self.weights = weights