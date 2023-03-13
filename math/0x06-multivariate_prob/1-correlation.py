#!/usr/bin/env python3
"""Correlation"""


def correlation(C):
    """Calculates a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a 2D numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    stdev = (np.diag(C) ** 0.5)
    D = np.diag(1 / stdev)
    corr_M = np.dot(np.dot(D, C), D)
    return corr_M
  
