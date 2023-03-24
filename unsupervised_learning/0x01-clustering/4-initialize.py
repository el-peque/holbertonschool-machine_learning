#!/usr/bin/env python3
"""Initialize GMM"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initializes variables for a Gaussian Mixture Model"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2\
       or not isinstance(k, int) or k <= 0 or k < X.shape[0]:
        return None
    pi = np.ones(k) / k
    m, _ = (kmeans(X, k))
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
