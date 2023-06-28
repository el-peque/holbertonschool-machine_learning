#!/usr/bin/env python3
"""Adjugate"""


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""
    cofactor = __import__('2-cofactor').cofactor
    cofactor_matrix = cofactor(matrix)
    adjugate_matrix = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(cofactor_matrix[j][i])
        adjugate_matrix.append(row)
    return adjugate_matrix
