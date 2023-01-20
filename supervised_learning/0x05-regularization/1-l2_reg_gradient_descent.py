#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network using
    gradient descent with L2 regularization"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        if i == L:
            dz = A - Y
        else:
            W2 = weights['W' + str(i + 1)]
            dz = np.multiply(np.matmul(W2.T, dz), 1 - (A ** 2))
        dW = (np.matmul(dz, cache['A' + str(i - 1)].T) + lambtha * W) / m
        db = (np.sum(dz, axis=1, keepdims=True) + lambtha * b) / m
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
