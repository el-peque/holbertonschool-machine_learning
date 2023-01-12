#!/usr/bin/env python3
"""Normalization Constants"""
import numpy as np


def normalization_constants(X):
    """Calculates the normalization constants of a matrix"""
    m = X.shape[0]
    mean = 1 / m * (np.sum(X, axis=0))
    variance = 1 / m * (np.sum(np.square(X - mean), axis=0))
    variance **= 0.5
    return mean, variance
