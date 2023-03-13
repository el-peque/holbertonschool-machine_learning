#!/usr/bin/env python3
"""Mean and Covariance"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.sum(X, axis=0) / n
    cov = np.matmul((X - mean).T, X - mean) / (n  - 1)
    return mean, cov
  
