#!/usr/bin/env python3
"""Slice Like A Ninja"""


def np_slice(matrix, axes={}):
    """slices a matrix along specific axes"""
    sl = [slice(None)] * matrix.ndim
    for k, v in axes.items():
        sl[k] = slice(*v)
    return matrix[tuple(sl)]
