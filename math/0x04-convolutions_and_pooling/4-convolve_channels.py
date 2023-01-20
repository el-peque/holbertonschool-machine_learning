#!/usr/bin/env python3
"""Convolution with Channels"""
import numpy as np
from time import sleep


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Performs a convolution on images with channels"""
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
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
    output = np.zeros((m, h, w))

    for i in range(h):
        for j in range(w):
            r = i * sh
            col = j * sw
            output[:, i, j] = np.sum(images[:, r:r+kh, col:col+kw, :] * kernel,
                                     axis=(1, 2, 3))
    return output
