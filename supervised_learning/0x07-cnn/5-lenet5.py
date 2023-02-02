#!/usr/bin/env python3
"""LeNet-5 (Keras)"""
import tensorflow.keras as K


def lenet5(X):
    """Builds a modified version of LeNet-5 architecture using tensorflow"""
    model = K.Sequential()
    # Convolutional layer with 6 kernels of shape 5x5 with same padding
    conv_1 = K.layers.Conv2D(filters=6,
                             kernel_size=(5, 5),
                             padding='same',
                             activation='relu',
                             kernel_initializer='he_normal')
    model.add(conv_1)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    maxpool_1 = K.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=(2, 2))
    model.add(maxpool_1)

    # Convolutional layer with 16 kernels of shape 5x5 with valid padding
    conv_2 = K.layers.Conv2D(filters=16,
                             kernel_size=(5, 5),
                             padding='valid',
                             activation='relu',
                             kernel_initializer='he_normal')
    model.add(conv_2)

    # Max pooling layer with kernels of shape 2x2 with 2x2 strides
    maxpool_2 = K.layers.MaxPool2D(pool_size=(2, 2),
                                   strides=(2, 2))
    flat = K.layers.Flatten()
    model.add(maxpool_2)
    model.add(flat)

    # Fully connected layer with 120 nodes
    fullycon_1 = K.layers.Dense(units=120,
                                activation='relu',
                                kernel_initializer='he_normal')
    model.add(fullycon_1)

    # Fully connected layer with 84 nodes
    fullycon_2 = K.layers.Dense(units=84,
                                activation='relu',
                                kernel_initializer='he_normal')
    model.add(fullycon_1)

    # Fully connected softmax output layer with 10 nodes
    softmax = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer='he_normal')
    model.add(softmax)
    model.compile(optimizer='Adam',
                  loss=K.losses.BinaryCrossentropy(),
                  metrics=['Accuracy'])

    return model
