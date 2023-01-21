#!/usr/bin/env python3
"""Create a Layer with Dropout"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """Creates a layer of a neural network using dropout"""
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg"))
    layer = tf.layers.Dense(units=n,
                            activation=activation,
                            kernel_initializer=init)(prev)
    dropout = tf.layers.Dropout(rate=keep_prob)(layer)
    return dropout
