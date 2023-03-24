#!/usr/bin/env python3
"""Optimize K"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance"""
    if not isinstance(X, np.ndarray) or not isinstance(kmin, int)\
       or not isinstance(kmax, int) or len(X.shape) != 2 or X.shape[0] < k\
       or k <= 0 or not isinstance(iterations, int) or iterations <= 0:
        return None, None
    results = []
    d_vars = []
    min_var = 0
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        var = variance(X, C)
        results.append((C, clss))
        if k == kmin:
            min_var = var
            d_vars.append(0)
        else:
            d_vars.append(min_var - var)

    return results, d_vars
