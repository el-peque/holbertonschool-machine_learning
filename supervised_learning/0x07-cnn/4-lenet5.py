#!/usr/bin/env python3
"""LeNet-5 (Tensorflow 1)"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """Builds a modified version of LeNet-5 architecture using tensorflow"""
    he_normal = tf.keras.initializers.VarianceScaling(scale=2.0)
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    lay_1 = tf.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)(x)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    lay_2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(lay_1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    lay_3 = tf.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation=tf.nn.relu,
                             kernel_initializer=he_normal)(lay_2)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    lay_4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(lay_3)
    lay_4_flat = tf.layers.Flatten()(lay_4)

    # Fully connected layer with 120 nodes
    lay_5 = tf.layers.Dense(units=120,
                            activation=tf.nn.relu,
                            kernel_initializer=he_normal)(lay_4_flat)

    # Fully connected layer with 84 nodes
    lay_6 = tf.layers.Dense(units=84,
                            activation=tf.nn.relu,
                            kernel_initializer=he_normal)(lay_5)

    # Fully connected softmax output layer with 10 nodes
    lay_7 = tf.layers.Dense(units=10,
                            kernel_initializer=he_normal)(lay_6)
    output = tf.nn.softmax(lay_7)

    loss = tf.losses.softmax_cross_entropy(y, logits=lay_7)

    equality = tf.math.equal(tf.argmax(y, axis=1), tf.argmax(lay_7, axis=1))
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    train_op = tf.train.AdamOptimizer().minimize(loss)

    return output, train_op, loss, accuracy
