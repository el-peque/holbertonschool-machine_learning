#!/usr/bin/env python3
"""Determinant"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or not all(isinstance(r,
                                               list) for r in matrix):
        raise TypeError('matrix must be a list of lists')

    n = len(matrix)
    if n == 0:
        raise TypeError('matrix must be a list of lists')
    if n == 1:
        if matrix[0] == []:
            return 1
    if any(len(row) != n for row in matrix):
        raise ValueError('matrix must be a square matrix')
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    total = 0
    sign = 1
    for i in range(n):
        M = [row[:i]+row[i+1:] for row in matrix[1:]]
        total += sign * matrix[0][i] * determinant(M)
        sign *= -1

    return total
