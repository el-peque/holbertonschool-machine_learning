#!/usr/bin/env python3
"""Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way"""
    X = np.random.permutation(X)
    Y = np.random.permutation(Y)
    return X, Y
