#!/usr/bin/env python3
"""Accuracy"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction"""
    y = tf.argmax(y)
    y_pred = tf.argmax(y_pred)
    equality = tf.math.equal(y_pred, y)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))
    return accuracy
