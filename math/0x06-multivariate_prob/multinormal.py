#!/usr/bin/env python3
"""Initialize"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""
    def __init__(self, data):
        """Instanciates class"""
        mean_cov = __import__('0-mean_cov').mean_cov

        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean, cov = mean_cov(data.T)
        self.mean, self.cov = mean.T, cov

    def pdf(self, x):
        """Calculates the PDF at a data point"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x must have the shape ({d}, 1)")
