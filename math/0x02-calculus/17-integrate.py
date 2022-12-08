#!/usr/bin/env python3
"""Integrate"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if not poly or not isinstance(poly, list):
        return None
    if not isinstance(C, int or float):
        return None

    integral = [0]
    for coeff in range(0, len(poly)):
        num = poly[coeff] / (coeff + 1)
        if num % 1 == 0:
            num = int(num)
        integral.append(num)

    return integral
