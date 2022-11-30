#!/usr/bin/env python3
"""add_arrays"""


def add_arrays(arr1, arr2):
    """Retunrs the sum of two arrays element-wise"""
    res = []
    if len(arr1) != len(arr2):
        return None
    for i in range(len(arr1)):
        res.insert(i, arr1[i] + arr2[i])

    return res
