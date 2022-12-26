#!/usr/bin/env python3
"""Class Neuron"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """Defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """nx is the number of input features to the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter method for W"""
        return self.__W

    @property
    def b(self):
        """Getter method for b"""
        return self.__b

    @property
    def A(self):
        """Getter method for A"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z = (np.matmul(self.W, X) + self.b)
        self.__A = 1/(1 + np.exp(-z))
        return self.A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -(np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A)) / m)
        return cost

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        pred = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return pred, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dZ = A - Y
        dW = (np.matmul(dZ, X.T)) / m
        self.__W = self.W - alpha * dW
        self.__b = np.sum(self.b - alpha * dZ) / m

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron"""
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
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if i % step == 0 or i == 0 or i == iterations:
                if verbose is True:
                    cost = self.cost(Y, A)
                    print(f"Cost after {i} iterations: {cost}")
                if graph is True:
                    costs.append(cost)
        pred, cost = self.evaluate(X, Y)
        if verbose is True:
            print(f"Cost after {i + 1} iterations: {cost}")
        if graph is True:
            costs.append(cost)
            plt.plot(np.arange(0, iterations + 1, step), costs, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training cost")
            plt.show()
        return pred, cost
