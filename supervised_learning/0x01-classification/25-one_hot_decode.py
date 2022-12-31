#!/usr/bin/env python3
"""one_hot_encode"""
import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray):
        return None
    decode_m = np.where(one_hot.T == 1)
    return decode_m[1]
