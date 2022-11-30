#!/usr/bin/env python3
"""cat_matrices2D"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    res = []
    if axis == 0:
        for i in mat1:
            if len(mat1[0]) != len(mat2[0]):
                return None
            res.append(i.copy())
        for i in mat2:
            res.append(i.copy())
    if axis == 1:
        for i in range(len(mat1)):
            res.append(mat1[i] + mat2[i])
    return res
