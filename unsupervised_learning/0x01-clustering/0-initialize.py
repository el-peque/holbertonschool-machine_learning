#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    min_x = np.min(X, axis=0)
    max_x = np.max(X, axis=0)
    centroids = np.random.uniform(min_x, max_x, size=(k, X.shape[1]))

    return centroids
