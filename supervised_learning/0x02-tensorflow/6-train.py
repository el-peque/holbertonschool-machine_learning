#!/usr/bin/env python3
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop
import tensorflow.compat.v1 as tf
"""Train"""


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, andº saves a neural network classifier"""
    for i in range(iterations):
        a = forward_prop(X_train, layer_sizes, activations)
        loss = calculate_loss(a, Y_train)
        #gradient_descent = create_train_op(loss, alpha)
        #weights = gradient_descent(Y_train, a)
        #with tf.Session() as sess:  print(loss.eval()) 
        #print(loss.value)

