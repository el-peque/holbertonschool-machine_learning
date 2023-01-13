#!/usr/bin/env python3
"""RMSProp Upgraded"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """Creates the training operation for a neural network
    in tensorflow using the RMSProp optimization algorithm"""
    train_op = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                         decay=beta2,
                                         epsilon=epsilon)
    return train_op.minimize(loss)
