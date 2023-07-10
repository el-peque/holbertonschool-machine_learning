#!/usr/bin/env python3
"""Principal components analysis"""
import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset"""
    n, d = X.shape
    covariance_matrix = np.dot(X.T, X) / n
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    explained_variance = np.cumsum(
        sorted_eigenvalues) / np.sum(sorted_eigenvalues)
    num_components = np.argmax(explained_variance >= var) + 1
    W = sorted_eigenvectors[:, :num_components] * -1

    return W
