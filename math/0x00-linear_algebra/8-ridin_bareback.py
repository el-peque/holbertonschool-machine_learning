#!/usr/bin/env python3
"""Matrix multiplication"""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None
    res = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            sum = 0
            for k in range(2):
                sum += mat1[i].copy()[k] * mat2[k].copy()[j]
            row.append(sum)
        res.append(row)
    return res.copy()
