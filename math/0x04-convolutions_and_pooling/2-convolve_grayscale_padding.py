#!/usr/bin/env python3
"""Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    h = h - kh + 2 * ph + 1
    w = w - kw + 2 * pw + 1
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)))
    output = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))
    return output
