#!/usr/bin/env python3
"""Learning Rate Decay Upgraded"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates a learning rate decay operation in tensorflow
    using inverse time decay"""
    op = tf.train.inverse_time_decay(learning_rate=alpha,
                                     global_step=global_step,
                                     decay_steps=decay_step,
                                     decay_rate=decay_rate,
                                     staircase=True)
    return op
