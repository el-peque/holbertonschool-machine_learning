#!/usr/bin/env python3
"""Multiple Kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """Performs a convolution on images using multiple kernels"""
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride
    ph, pw = 0, 0

    if isinstance(padding, tuple):
        ph, pw = padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1

    h = (h - kh + 2 * ph) // sh + 1
    w = (w - kw + 2 * pw) // sw + 1
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    output = np.zeros((m, h, w, nc))

    for k in range(nc):
        for i in range(h):
            for j in range(w):
                r = i * sh
                col = j * sw
                output[:, i, j, k] = np.sum(images[:, r:r+kh, col:col+kw, :] *
                                            kernels[:, :, :, k],
                                            axis=(1, 2, 3))
    return output
