#!/usr/bin/env python3
"""Adam Upgraded"""
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network
    in tensorflow using the Adam optimization algorithm"""
    train_op = tf.train.AdamOptimizer(learning_rate=alpha,
                                      beta1=beta1,
                                      beta2=beta2,
                                      epsilon=epsilon)
    return train_op.minimize(loss)
