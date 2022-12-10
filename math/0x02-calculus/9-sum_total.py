#!/usr/bin/env python3
"""summation_i_squared"""


def summation_i_squared(n):
    """calculates the sum of the squares of
    the first n positive integers"""
    if not isinstance(n, int) or n < 1:
        return None
    if n == 1:
        return 1
    sum = (n**2 + summation_i_squared(n - 1))
    return sum
