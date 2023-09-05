#!/usr/bin/env python3
"""Bidirectional Cell Backward"""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """Class constructor"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step"""
        concat_input_f = np.hstack((h_prev, x_t))
        h_next_f = np.tanh(np.dot(concat_input_f, self.Whf) + self.bhf)

        return h_next_f

    def backward(self, h_next, x_t):
        """Performs backward propagation for one time step"""
        concat_input_b = np.hstack((h_next, x_t))
        h_prev_b = np.tanh(np.dot(concat_input_b, self.Whb) + self.bhb)

        return h_prev_b
