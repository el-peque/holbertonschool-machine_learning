#!/usr/bin/env python3
"""Inception Block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Builds an inception block"""
    print(filters)
    conv1 = K.layers.Conv2D(filters=filters[0],
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(A_prev)

    conv2 = K.layers.Conv2D(filters=filters[1],
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(A_prev)
    conv2 = K.layers.Conv2D(filters=filters[2],
                            kernel_size=(3, 3),
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(conv2)

    conv3 = K.layers.Conv2D(filters=filters[3],
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(A_prev)
    conv3 = K.layers.Conv2D(filters=filters[4],
                            kernel_size=(5, 5),
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(conv3)

    maxpool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    padding='same',
                                    strides=(1, 1))(A_prev)
    conv4 = K.layers.Conv2D(filters=filters[5],
                            kernel_size=(1, 1),
                            activation='relu',
                            padding='same',
                            kernel_initializer='he_normal')(maxpool)

    concatted = K.layers.concatenate([conv1, conv2, conv3, conv4])
    return concatted
