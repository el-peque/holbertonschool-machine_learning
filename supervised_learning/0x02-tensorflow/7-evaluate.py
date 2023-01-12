#!/usr/bin/env python3
"""Evaluate"""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network"""
    saver = tf.train.import_meta_graph(save_path + ".meta")

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[-1]
        y = tf.get_collection('y')[-1]
        y_pred = tf.get_collection('y_pred')[-1]
        accuracy = tf.get_collection('accuracy')[-1]
        loss = tf.get_collection('loss')[-1]
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})
        accuracy_value = sess.run(accuracy, feed_dict={x: X, y: Y})
        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
    return pred, accuracy_value, loss_value
