#!/usr/bin/env python3
"""The Whole Barn"""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    mat1_shape, mat2_shape = matrix_shape(mat1), matrix_shape(mat2)
    if not mat1_shape or not mat2_shape or mat1_shape != mat2_shape:
        return None
    mat3 = []
    if len(mat1_shape) == 1:
        for i in range(len(mat1)):
            mat3.append(mat1[i] + mat2[i])
    else:
        for i in range(len(mat1)):
            mat3.append(add_matrices(mat1[i], mat2[i]))
    return mat3
