#!/usr/bin/env python3
"""Variance"""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2\
       or not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    try:
        distances = np.sqrt(((X - C[:, np.newaxis])**2).sum(axis=2))
        var = np.sum(np.min(distances, axis=0)**2)
        return var
    except Exception:
        return None 
