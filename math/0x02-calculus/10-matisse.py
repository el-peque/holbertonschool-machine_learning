#!/usr/bin/env python3
"""poly_derivative"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial"""
    if not poly or not isinstance(poly, list):
        return None
    if len(poly) == 1:
        return [0]

    derivative = []
    for coeff in range(1, len(poly)):
        derivative.append((coeff) * poly[coeff])

    return derivative
