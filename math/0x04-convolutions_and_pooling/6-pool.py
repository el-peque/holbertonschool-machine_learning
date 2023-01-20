#!/usr/bin/env python3
"""Pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Performs pooling on images"""
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    h = (h - kh) // sh + 1
    w = (w - kw) // sw + 1
    output = np.zeros((m, h, w, c))

    for i in range(h):
        for j in range(w):
            r = i * sh
            col = j * sw
            if mode == 'max':
                output[:, i, j] = np.max(images[:, r:r+kh, col:col+kw],
                                         axis=(1, 2))
            if mode == 'avg':
                output[:, i, j] = np.average(images[:, r:r+kh, col:col+kw],
                                             axis=(1, 2))
    return output
