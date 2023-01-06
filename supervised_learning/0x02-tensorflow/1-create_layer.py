#!/usr/bin/env python3
"""Layers"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(units=n, kernel_initializer=initializer,
                                  activation=activation, name='layer')(prev)
    return layer
