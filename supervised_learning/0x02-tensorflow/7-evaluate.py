#!/usr/bin/env python3
"""Evaluate"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + ".meta")
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        accuracy_value = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})

        return pred, accuracy_value, loss_value
