#!/usr/bin/env python3
"""K-Means"""
import numpy as np
initialize = __import__('0-initialize').initialize


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or not isinstance(k, int)\
       or len(X.shape) != 2 or X.shape[0] < k or k <= 0 or iterations <= 0:
        return None
    C = initialize(X, k)
    clss = np.zeros(X.shape[0],)

    for i in range(iterations):
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
        clss = np.argmin(distances, axis=0)
        prev_C = C.copy()
        for j in range(k):
            if np.sum(clss == j) == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.allclose(C, prev_C):
            return C, clss

    return None, None
