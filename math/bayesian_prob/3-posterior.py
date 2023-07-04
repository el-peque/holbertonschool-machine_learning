#!/usr/bin/env python3
"""Posterior"""
import numpy as np


def posterior(x, n, P, Pr):
    """Calculates the posterior probability for the various hypothetical
    probabilities of developing severe side effects given the data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
            )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    nCx = np.math.factorial(n)/(np.math.factorial(x)*np.math.factorial(n-x))
    likelihood = P.copy()
    for i, p in enumerate(P):
        likelihood[i] = nCx * (p ** x) * ((1 - p) ** (n - x))
    intersection = likelihood * Pr

    return intersection / np.sum(intersection)
