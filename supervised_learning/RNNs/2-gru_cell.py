#!/usr/bin/env python3
"""GRU Cell"""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """
        Class constructor:
          i is the dimensionality of the data
          h is the dimensionality of the hidden state
          o is the dimensionality of the outputs
        """
        self.Wz = np.random.normal(size=(h+i, h))
        self.bz = np.zeros((1, h))

        self.Wr = np.random.normal(size=(h+i, h))
        self.br = np.zeros((1, h))

        self.Wh = np.random.normal(size=(h+i, h))
        self.bh = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step"""
        concat_input = np.concatenate((h_prev, x_t), axis=-1)

        z = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wz) + self.bz)))

        r = 1 / (1 + np.exp(-(np.dot(concat_input, self.Wr) + self.br)))

        concat_reset_input = np.hstack((r * h_prev, x_t))
        h_candidate = np.tanh(np.dot(concat_reset_input, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_candidate

        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, y
