#!/usr/bin/env python3
"""Calculate Loss"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(y, logits=y_pred)
    return loss
