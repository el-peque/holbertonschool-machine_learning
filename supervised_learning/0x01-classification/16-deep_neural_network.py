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
        # if not all(isinstance(i, int) or i > 0 for i in layers):
        #     raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        weights = {}
        for i, layer in enumerate(layers):
            if layer < 1:
                raise TypeError("layers must be a list of positive integers")
            b = np.zeros((layer, 1))
            if i == 0:
                weight = np.random.randn(layer, nx) * np.sqrt(2 / nx)
            else:
                f1 = np.random.randn(layer, layers[i - 1])
                f2 = np.sqrt(2 / layers[i - 1])
                weight = f1 * f2
            weights["W" + str(i + 1)] = weight
            weights["b" + str(i + 1)] = b
        self.weights = weights
