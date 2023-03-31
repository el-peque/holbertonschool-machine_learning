#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """Determines the probability of a markov chain being in a
    particular state after a specified number of iterations"""
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1] != s.shape[1]:
        return
    if not all(np.sum(P, axis=1) == 1):
        return
    if not isinstance(s, np.ndarray) or s.shape[0] != 1 or t < 1:
        return

    for i in range(t):
        s = s @ P
    return s
