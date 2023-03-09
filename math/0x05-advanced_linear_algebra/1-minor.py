#!/usr/bin/env python3
""" Minor """


def minor(matrix):
    """Calculates the minor matrix of a matrix"""
    determinant = __import__('0-determinant').determinant

    if not isinstance(matrix, list) or not all(isinstance(r,
                                               list) for r in matrix):
        raise TypeError('matrix must be a list of lists')
    n = len(matrix)
    if n == 0:
        raise TypeError('matrix must be a list of lists')
    if n == 1 == len(matrix[0]):
        return [[1]]
    if any(len(row) != n for row in matrix):
        raise ValueError('matrix must be a non-empty square matrix')

    minor_matrix = []
    for i in range(n):
        minor_row = []
        for j in range(len(matrix[i])):
            sub_matrix = [row[:j] + row[j+1:] for row in matrix[:i] +
                          matrix[i+1:]]
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix
