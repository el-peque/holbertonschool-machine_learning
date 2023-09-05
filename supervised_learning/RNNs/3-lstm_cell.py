#!/usr/bin/env python3
"""LSTM Cell"""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))

        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))

        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))

        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))

        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step"""

        concat_input = np.hstack((h_prev, x_t))

        forget_gate = 1 / \
            (1 + np.exp(-(np.dot(concat_input, self.Wf) + self.bf)))

        update_gate = 1 / \
            (1 + np.exp(-(np.dot(concat_input, self.Wu) + self.bu)))

        c_candidate = np.tanh(np.dot(concat_input, self.Wc) + self.bc)

        c_next = forget_gate * c_prev + update_gate * c_candidate

        output_gate = 1 / \
            (1 + np.exp(-(np.dot(concat_input, self.Wo) + self.bo)))

        h_next = output_gate * np.tanh(c_next)

        y = np.exp(np.dot(h_next, self.Wy) + self.by)
        y = y / np.sum(y, axis=1, keepdims=True)

        return h_next, c_next, y
