#!/usr/bin/env python3
"""one_hot_encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray):
        return None
    if not isinstance(classes, int) or classes < np.amax(Y):
        return None
    one_hot_m = np.zeros(shape=(len(Y), classes))
    for column, row in enumerate(Y):
        one_hot_m[column, row] = 1
    return one_hot_m.T
