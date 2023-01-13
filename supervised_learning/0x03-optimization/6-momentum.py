#!/usr/bin/env python3
"""Momentum Upgraded"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """Creates the training operation for a neural network
    in tensorflow using the gradient descent with
    momentum optimization algorithm"""
    train_op = tf.train.MomentumOptimizer(learning_rate=alpha, momentum=beta1)
    return train_op.minimize(loss)
