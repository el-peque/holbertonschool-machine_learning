#!/usr/bin/env python3
"""Convolutional Forward Prop"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    ph, pw = 0, 0

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    h = (h_prev - kh + 2 * ph) // sh + 1
    w = (w_prev - kw + 2 * pw) // sw + 1
    A = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    output = np.zeros((m, h, w, c_new))
    for i in range(h):
        for j in range(w):
            for k in range(c_new):
                r = i * sh
                col = j * sw
                Z = np.sum(A[:, r:r+kh, col:col+kw, :] * W[:, :, :, k],
                           axis=(1, 2, 3))
                output[:, i, j, k] = Z
    Z = output + b
    return activation(Z)
