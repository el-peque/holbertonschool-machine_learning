#!/usr/bin/env python3
"""Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN"""
    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t, m, 2 * h))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    Hf = np.zeros((t, m, h))
    Hb = np.zeros((t, m, h))

    for step in range(t):
        x_t = X[step]
        h_0 = bi_cell.forward(h_0, x_t)
        h_t = bi_cell.backward(h_t, x_t)

        Hf[step] = h_0
        Hb[t - 1 - step] = h_t

    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
