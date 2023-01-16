#!/usr/bin/env python3
"""Batch Normalization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """Normalizes an unactivated output of a neural network
    using batch normalization"""
    m = Z.shape[0]
    mean = (np.sum(Z, axis=0)) / m
    variance = (np.sum(np.square(Z - mean), axis=0)) / m
    Z_norm = (Z - mean) / ((variance + epsilon) ** 0.5)
    Z_norm = gamma * Z_norm + beta
    return Z_norm
