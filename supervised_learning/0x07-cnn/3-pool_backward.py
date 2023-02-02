#!/usr/bin/env python3
"""Pooling Back Prop"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Performs back propagation over a pooling layer of a neural network"""
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output = np.zeros(A_prev.shape)
    for f in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c):
                    r = i * sh
                    col = j * sw
                    if mode == 'avg':
                        avg_dA = dA[f, i, j, k] / (kh * kw)
                        output[f, r:r+kh, col:col+kw, k] += np.multiply(
                                                             np.ones((kh, kw)),
                                                             avg_dA)
                    if mode == 'max':
                        a_prev_slice = A_prev[f, r:r+kh, col:col+kw, k]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        output[f, r:(r+kh), col:(col+kw), k] += np.multiply(
                                                                mask,
                                                                dA[f, i, j, k])
    return output
