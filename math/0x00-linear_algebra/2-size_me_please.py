#!/usr/bin/env python3
"""matrix_shape"""


def matrix_shape(matrix):
    """Returns the shape of a matrix"""
    shape = []
    if matrix and isinstance(matrix, list):
        matrix_cpy = matrix.copy()
        shape.append(len(matrix_cpy))

    while isinstance(matrix_cpy[0], list):
        matrix_cpy = matrix_cpy[0]
        shape.append(len(matrix_cpy))

    return shape
