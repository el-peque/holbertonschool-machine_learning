#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """Updates a variable in place using the Adam optimization algorithm"""
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    v_hat = v / (1 - beta1 ** t)
    s_hat = s / (1 - beta2 ** t)
    var = var - (alpha * (v_hat / (s_hat ** 0.5 + epsilon)))
    return var, v, s
