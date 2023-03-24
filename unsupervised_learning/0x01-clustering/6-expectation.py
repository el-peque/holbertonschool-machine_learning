#!/usr/bin/env python3
"""Expectation"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""
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
