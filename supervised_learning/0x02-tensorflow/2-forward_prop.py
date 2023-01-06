#!/usr/bin/env python3
"""Forward Propagation"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural network"""
    y = x
    for size, activation in zip(layer_sizes, activations):
        y = create_layer(y, size, activation)
    return y
