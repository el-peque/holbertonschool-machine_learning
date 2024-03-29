#!/usr/bin/env python3
"""Class DeepNeuralNetwork"""
import matplotlib.pyplot as plt
import numpy as np
import pickle


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers, activation='sig'):
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
        if activation not in ('sig', 'tanh'):
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__activation = activation
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

    @property
    def activation(self):
        """Returns the activation functiona used in the hidden layers"""
        return self.__activation

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache["A0"] = X
        for i in range(self.L):
            w = self.weights['W' + str(i + 1)]
            b = self.weights['b' + str(i + 1)]
            if i == 0:
                z = np.matmul(w, X) + b
            else:
                z = np.matmul(w, A) + b
            if self.activation == 'sig':
                A = 1 / (1 + np.exp(-z))
            if self.activation == 'tanh':
                A = 2 / (1 + (np.exp(-2 * z))) - 1
            if i == self.L - 1:
                A = np.exp(z) / np.sum(np.exp(z), axis=0)
            self.__cache['A' + str(i + 1)] = A
        return A, self.cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = - np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, cache = self.forward_prop(X)
        classes = Y.shape[0]
        pred = DeepNeuralNetwork.one_hot_encode(np.argmax(A, axis=0), classes)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]
        weights = self.weights.copy()
        for i in range(self.L, 0, -1):
            A = cache['A' + str(i)]
            if i == self.L:
                dz = A - Y
            elif self.activation == 'sig':
                dz = np.multiply(np.matmul(
                     (weights['W' + str(i + 1)]).T, dz),
                     A * (1 - A))
            elif self.activation == 'tanh':
                dz = np.multiply(np.matmul(
                     (weights['W' + str(i + 1)]).T, dz),
                     1 - (A ** 2))
            dW = np.matmul(dz, cache['A' + str(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
            self.__weights['b' + str(i)] = weights['b' + str(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the deep neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if i % step == 0 or i == 0 and (verbose or graph):
                cost = self.cost(Y, A)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
                if graph is True:
                    costs.append(cost)

        pred, cost = self.evaluate(X, Y)
        if verbose is True:
            print("Cost after {} iterations: {}".format(i + 1, cost))
        if graph is True:
            costs.append(cost)
            plt.plot(np.arange(0, iterations + 1, step), costs, "b-")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return pred, cost

    def save(self, filename):
        """Saves the instance object to a file in pickle format"""
        if not isinstance(filename, str):
            return None
        if filename[-4:] != '.pkl':
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object"""
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
            return obj
        except FileNotFoundError:
            return None

    @staticmethod
    def one_hot_encode(Y, classes):
        """Converts a numeric label vector into a one-hot matrix"""
        if not isinstance(Y, np.ndarray):
            return None
        if not isinstance(classes, int) or classes < np.amax(Y):
            return None
        one_hot_m = np.zeros(shape=(len(Y), classes))
        for column, row in enumerate(Y):
            one_hot_m[column, row] = 1
        return one_hot_m.T
