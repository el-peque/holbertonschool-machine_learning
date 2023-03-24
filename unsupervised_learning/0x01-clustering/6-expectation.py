#!/usr/bin/env python3
"""Expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""
    if not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray)\
       or not isinstance(S, np.ndarray) or not isinstance(pi, np.ndarray)\
       or len(X.shape) != 2 or len(pi.shape) != 1 or len(m.shape) != 2\
       or len(S.shape) != 3\
       or not (X.shape[1] == m.shape[1] == S.shape[1] == S.shape[2])\
       or not (pi.shape[0] == m.shape[0] == S.shape[0]):
        return None, None
    try:
        dens = np.zeros((pi.shape[0], X.shape[0]))
        for i in range(pi.shape[0]):
            dens[i] = pdf(X, m[i], S[i])

        numerator = np.multiply(dens.T, pi)
        denominator = np.sum(numerator, axis=1)
        g = numerator / denominator[:, np.newaxis]
        l = np.sum(np.log(denominator))

        return g, l
    except Exception:
        return None, None
