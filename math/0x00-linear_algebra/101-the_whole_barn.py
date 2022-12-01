#!/usr/bin/env python3
"""The Whole Barn"""
matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices(mat1, mat2):
    """Adds two matrices"""
    mat1_shape, mat2_shape = matrix_shape(mat1), matrix_shape(mat2)
    if not mat1_shape or not mat2_shape or mat1_shape != mat2_shape:
        return None
    mat3 = mat1.copy()
    for i in range(mat1_shape[0]):
        if len(mat1_shape) > 1:
            for j in range(mat1_shape[1]):
                if len(mat1_shape) > 2:
                    for k in range(mat1_shape[2]):
                        mat3[i][j][k] += mat2.copy()[i][j][k]
                else:
                    mat3[i][j] += mat2.copy()[i][j]
        else:
            mat3[i] += mat2.copy()[i]

    return mat3
