#!/usr/bin/env python3
"""L2 Regularization Cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost of a neural network with L2 regularization"""
    w = 0
    for weight in weights.values():
        w += np.sum(np.square(weight))
    cost = cost + (lambtha / (2 * m)) * w
    return cost
