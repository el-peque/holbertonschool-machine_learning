#!/usr/bin/env python3
"""Regular Chains"""
import numpy as np


def regular(P):
    """Determines the steady state probabilities of a regular markov chain"""
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return
    if not all(np.sum(P, axis=1) == 1):
        return
    try:
        p = P
        dim = p.shape[0]
        q = (p-np.eye(dim))
        ones = np.ones(dim)
        q = np.c_[q,ones]
        QTQ = np.dot(q, q.T)
        bQT = np.ones(dim)
        return np.linalg.solve(QTQ,bQT)
    except Exception:
        return
