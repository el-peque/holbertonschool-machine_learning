#!/usr/bin/env python3
"""Convolution with Padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding.shape
    pad_top = ph // 2
    pad_bottom = ph - pad_top
    pad_left = pw // 2
    pad_right = pw - pad_left
    images = np.pad(images, ((0, 0),
                             (pad_top, pad_bottom),
                             (pad_left, pad_right)))
    output = np.zeros(shape=(m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))
    return output
