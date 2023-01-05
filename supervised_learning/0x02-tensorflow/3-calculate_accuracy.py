#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""Accuracy"""


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
