#!/usr/bin/env python3
"""poly_derivative"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if not poly or not isinstance(poly, list):
        return None

    derivative = []
    for coeff in range(len(poly), 1, -1):
        derivative.append((coeff - 1) * poly[coeff - 1])

    if all(x == 0 for x in derivative):
        return [0]

    return derivative
