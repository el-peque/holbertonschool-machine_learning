#!/usr/bin/env python3
"""add_matrices2D"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""
    res = []
    if len(mat1) != len(mat2):
        return None
    for i in range(len(mat1)):
        row = []
        if len(mat1[i]) != len(mat2[i]):
            return None
        for j in range(len(mat1[i])):
            row.insert(j, mat1[i][j] + mat2[i][j])
        res.append(row)

    return res
