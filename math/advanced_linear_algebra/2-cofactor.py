#!/usr/bin/env python3
"""Cofactor"""


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix"""
    minor = __import__('1-minor').minor

    minor_matrix = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            sign = (-1) ** (i + j)
            minor_matrix[i][j] *= sign
    return minor_matrix
