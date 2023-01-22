#!/usr/bin/env python3
"""Gradient Descent with Dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates the weights of a neural network with
    Dropout regularization using gradient descent"""
    m = Y.shape[1]
    w = weights.copy()
    for i in range(L, 0, -1):
        A = cache['A' + str(i)]
        if i == L:
            dz = A - Y
        else:
            dz = np.multiply(np.matmul((w['W' + str(i + 1)]).T, dz), (1 - (A ** 2)))
            D = cache['D' + str(i)]
            dz = (dz * D) / keep_prob
        dW = np.matmul(dz, cache['A' + str(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W' + str(i)] = w['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = w['b' + str(i)] - alpha * db
