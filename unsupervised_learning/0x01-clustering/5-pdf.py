#!/usr/bin/env python3
"""Probability Density Function"""
import numpy as np


def pdf(X, m, S):
    """Calculates the PDF of a Gaussian distribution"""
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray)\
       or not isinstance(S, np.ndarray) or len(X.shape) != 2 or\
       len(m.shape[0]) != 1 or len(S.shape) != 2 or\
       not (X.shape[1] == m.shape[0] == S.shape[0] == S.shape[1]):
        None
    d = m.shape[0]
    X = X - m.reshape(1, d)
    P = np.exp(-0.5 * np.sum(X @ np.linalg.inv(S) * X, axis=1)) /\
        np.sqrt((2 * np.pi)**d * np.linalg.det(S))
    P = np.maximum(P, 1e-300)
    return P
