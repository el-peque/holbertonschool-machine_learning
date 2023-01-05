#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""Layers"""


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, bias_initializer=initializer,
                                  activation=activation, name='layer')(prev)
    return layer
