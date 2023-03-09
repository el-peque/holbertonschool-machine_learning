#!/usr/bin/env python3
"""Definiteness"""
import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    try:
        if np.all(np.linalg.eigvals(matrix) > 0):
            return "Positive definite"
        if np.all(np.linalg.eigvals(matrix) < 0):
            return "Negative definite"
        if np.all(np.linalg.eigvals(matrix) >= 0):
            return "Positive semi-definite"
        if np.all(np.linalg.eigvals(matrix) <= 0):
            return "Negative semi-definite"
        return "Indefinite"

    except np.linalg.LinAlgError as err:
        return
