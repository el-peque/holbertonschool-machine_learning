#!/usr/bin/env python3
"""RNN Cell"""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        Class constructor:
          i is the dimensionality of the data
          h is the dimensionality of the hidden state
          o is the dimensionality of the outputs
        """
        self.Wh = np.random.normal(size=(h, h + i))
        self.Wy = np.random.normal(size=(o, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step"""
        concat_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh((concat_input @ self.Wh.T) + self.bh)
        y = (h_next @ self.Wy.T) + self.by
        return h_next, y
