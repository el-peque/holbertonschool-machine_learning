#!/usr/bin/env python3
"""cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    res = []
    if axis == 0:
        return mat1 + mat2
    for i in range(len(mat1)):
        res.append(mat1[i] + mat2[i])
    return res
