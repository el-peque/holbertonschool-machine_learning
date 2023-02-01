#!/usr/bin/env python3
"""Pooling Forward Prop"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h = (h_prev - kh) // sh + 1
    w = (w_prev - kw) // sw + 1
    output = np.zeros((m, h, w, c_prev))
    for i in range(h):
        for j in range(w):
            r = i * sh
            col = j * sw
            if mode == 'max':
                output[:, i, j] = np.max(A_prev[:, r:r+kh, col:col+kw],
                                         axis=(1, 2))
            if mode == 'avg':
                output[:, i, j] = np.average(A_prev[:, r:r+kh, col:col+kw],
                                             axis=(1, 2))
    return output
