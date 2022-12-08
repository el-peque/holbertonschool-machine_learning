#!/usr/bin/env python3
"""summation_i_squared"""


def summation_i_squared(n):
    """calculates the sum of the squares of
    the first n positive integers"""
    if not n or not isinstance(n, int) or n < 0:
        return None
    if n == 0:
        return 0
    return (n**2 + summation_i_squared(n - 1))
