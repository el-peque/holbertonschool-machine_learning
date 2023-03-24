#!/usr/bin/env python3
"""K-Means"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or not isinstance(k, int)\
       or len(X.shape) != 2 or X.shape[0] < k or k <= 0 or iterations <= 0:
        return None, None
    try:
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        C = np.random.uniform(low=min_vals, high=max_vals,
                              size=(k, X.shape[1]))
        clss = np.zeros(X.shape[0],)
        for i in range(iterations):
            distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
            clss = np.argmin(distances, axis=0)
            old_C = np.copy(C)
            for j in range(k):
                if np.sum(clss == j) == 0:
                    C[j] = np.random.uniform(low=min_vals, high=max_vals,
                                             size=(1, X.shape[1]))
                else:
                    C[j] = np.mean(X[clss == j], axis=0)
            if np.allclose(C, old_C):
                return C, clss
        return None, None
    except Exception:
        return None, None
