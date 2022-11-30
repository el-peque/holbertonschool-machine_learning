#!/usr/bin/env python3

def matrix_shape(matrix):
    shape = []
    matrix_cpy = matrix.copy()
    if isinstance(matrix_cpy, list):
        shape.append(len(matrix_cpy))

    while isinstance(matrix_cpy[0], list):
        matrix_cpy = matrix_cpy[0]
        shape.append(len(matrix_cpy))

    return shape
