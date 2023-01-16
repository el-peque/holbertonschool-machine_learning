#!/usr/bin/env python3
"""Batch Normalization Upgraded"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural net in tensorflow"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    X = tf.keras.layers.Dense(units=n, kernel_initializer=initializer,
                              activation=activation)(prev)
    offset = tf.Variable(tf.zeros(n), dtype=tf.float32)
    scale = tf.Variable(tf.ones(n), dtype=tf.float32)
    mean, variance = tf.nn.moments(X, axes=[0])
    batch_norm = tf.nn.batch_normalization(x=X,
                                           mean=mean,
                                           variance=variance,
                                           offset=offset,
                                           scale=scale,
                                           variance_epsilon=1e-8)
    return batch_norm
