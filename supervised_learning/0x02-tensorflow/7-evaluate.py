#!/usr/bin/env python3
"""Evaluate"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(save_path + ".meta")

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        x = tf.compat.v1.get_collection('x')[0]
        y = tf.compat.v1.get_collection('y')[0]
        y_pred = tf.compat.v1.get_collection('y_pred')[0]
        accuracy = tf.compat.v1.get_collection('accuracy')[0]
        loss = tf.compat.v1.get_collection('loss')[0]
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy_value = sess.run(accuracy, feed_dict={x: X, y: Y})
        predictions = sess.run(y_pred, feed_dict={x: X})
    return predictions, accuracy_value, loss_value
