#!/usr/bin/env python3
"""Absorbing Chains"""
import numpy as np


def absorbing(P):
    """Determines if a markov chain is absorbing"""
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return False
    if not all(np.sum(P, axis=1) == 1):
        return False
    try:
        if not any(np.diag(P) == 1):
            return False
        if all(np.diag(P) == 1):
            return True
        return True
    except Exception:
        return False
