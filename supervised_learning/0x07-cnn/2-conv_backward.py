#!/usr/bin/env python3
"""Convolutional Back Prop"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    weights = W.copy()
    ph, pw = 0, 0

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1

    h = (h_prev - kh + 2 * ph) // sh + 1
    w = (w_prev - kw + 2 * pw) // sw + 1

    dA = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    dW = np.zeros(W.shape)

    for f in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    r = i * sh
                    col = j * sw
                    dA[f, r:r+kh, col:col+kw, :] += np.multiply(
                                                        weights[:, :, :, k],
                                                        dZ[f, i, j, k]
                                                        )
                    dW[:, :, :, k] += np.multiply(
                                          A_prev[f, r:r+kh, col:col+kw, :],
                                          dZ[f, i, j, k]
                                          )

    return dA, dW, db
