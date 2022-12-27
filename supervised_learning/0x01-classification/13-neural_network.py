#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """Defines a neural network with one hidden
       layer performing binary classification"""
    def __init__(self, nx, nodes):
        """
        nx is the number of input features to the neuron
        nodes is the number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter method for W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter method for b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter method for A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter method for W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter method for b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter method for A2"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1/(1 + np.exp(-z1))
        z2 = np.matmul(self.W2, self.A1) + self.b2
        self.__A2 = 1/(1 + np.exp(-z2))
        return self.A1, self.A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A1, A2 = self.forward_prop(X)
        pred = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return pred, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        A1, A2 = self.forward_prop(X)
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.multiply(np.matmul((self.W2).T, dZ2), A1 * (1 - A1))
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W1 = self.W1 - alpha * dW1
        self.__b1 = self.b1 - alpha * db1
        self.__W2 = self.W2 - alpha * dW2
        self.__b2 = self.b2 - alpha * db2
