#!/usr/bin/env python3
"""Bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """Performs forward propagation for a bidirectional RNN"""
    t, m, i = X.shape
    h = h_0.shape[1]

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))
    H_concatenated = np.zeros((t, m, 2 * h))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    h_prev_forward = h_0
    h_prev_backward = h_t

    for step in range(t):
        x_t = X[step]

        h_next_forward = bi_cell.forward(h_prev_forward, x_t)
        H_forward[step] = h_next_forward
        h_prev_forward = h_next_forward

        x_t_backward = np.flip(x_t, axis=0)
        h_next_backward = bi_cell.backward(h_prev_backward, x_t_backward)
        H_backward[step] = h_next_backward
        h_prev_backward = h_next_backward

        H_concatenated[step] = np.hstack((h_next_forward, h_next_backward))

        Y[step] = bi_cell.output(H_concatenated[step])

    return H_concatenated, Y
