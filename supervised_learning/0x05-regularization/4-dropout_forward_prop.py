#!/usr/bin/env python3
"""Forward Propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout"""
    cache = {'A0': X}
    for i in range(L):
        w = weights['W' + str(i + 1)]
        b = weights['b' + str(i + 1)]
        if i == 0:
            z = np.matmul(w, X) + b
        else:
            z = np.matmul(w, A) + b
        if i == L - 1:
            A = np.exp(z) / np.sum(np.exp(z), axis=0)
        else:
            A = np.tanh(z)
            D = np.random.randn(A.shape[0], A.shape[1])
            D = D < keep_prob
            A = (np.multiply(A, D)) / keep_prob
        cache['A' + str(i + 1)] = A
    return cache
