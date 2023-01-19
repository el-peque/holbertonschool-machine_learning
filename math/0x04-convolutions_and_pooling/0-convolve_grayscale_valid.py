#!/usr/bin/env python3
"""Valid Convolution"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    output = np.zeros(shape=(m, h - kh + 1, w - kw + 1))
    for i in range(output.shape[1]):
        for j in range(output.shape[2]):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel,
                                     axis=(1, 2))
    return output
