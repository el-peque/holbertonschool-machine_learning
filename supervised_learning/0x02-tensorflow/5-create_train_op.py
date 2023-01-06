#!/usr/bin/env python3
"""Train_Op"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network"""
    gd = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    return gd.minimize(loss)
