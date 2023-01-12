#!/usr/bin/env python3
"""Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    p = np.random.permutation(len(X))
    return X[p], Y[p]
