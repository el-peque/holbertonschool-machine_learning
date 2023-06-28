#!/usr/bin/env python3
"""Inverse"""


def inverse(matrix):
    """Calculates the inverse of a matrix"""
    determinant = __import__('0-determinant').determinant
    adjugate = __import__('3-adjugate').adjugate
    m_adjugate = adjugate(matrix)
    m_determinant = determinant(matrix)
    if m_determinant == 0:
        return
    m_inverse = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[i])):
            row.append(m_adjugate[i][j] / m_determinant)
        m_inverse.append(row)
    return m_inverse
