#!/usr/bin/env python3
"""Strided Convolution"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride
    ph, pw = 0, 0

    if isinstance(padding, tuple):
        ph, pw = padding
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1

    h = (h - kh + 2 * ph) // sh + 1
    w = (w - kw + 2 * pw) // sw + 1
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            r = i * sh
            c = j * sw
            output[:, i, j] = np.sum(images[:, r:r+kh, c:c+kw] * kernel,
                                     axis=(1, 2))
    return output
