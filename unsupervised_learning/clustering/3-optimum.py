#!/usr/bin/env python3
"""Optimize K"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """Tests for the optimum number of clusters by variance"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmin, int) or not isinstance(kmax, int)\
       or not isinstance(iterations, int):
        return None, None
    if kmin < 1 or kmax < kmin or iterations < 1:
        return None, None

    try:
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
    except Exception:
        return None, None
