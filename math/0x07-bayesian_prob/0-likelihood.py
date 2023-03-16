#!/usr/bin/env python3
"""Likelihood"""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects"""
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
    if max(P) > 1 or min(P) < 0:
        raise ValueError("All values in P must be in the range [0, 1]")

    nCx = np.math.factorial(n) / (np.math.factorial(x) * np.math.factorial(n-x))
    likelihood = P.copy()
    for i, p in enumerate(P):
        likelihood[i] = nCx * (p ** x) * ((1 - p) ** (n - x))
    return likelihood
