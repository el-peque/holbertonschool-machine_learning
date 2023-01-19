#!/usr/bin/env python3
"""Same Convolution"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """Performs a same convolution on grayscale images"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    pad_h = np.max(kh, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_w = np.max(kw, 0)
    pad_left = pad_w // 2
    pad_right = pad_h - pad_left
    images = np.pad(images, ((0, 0),
                             (pad_top, pad_bottom),
                             (pad_left, pad_right)))
    output = np.zeros(shape=(m, h, w))
    for i in range(h):
        for j in range(w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))
    return output
