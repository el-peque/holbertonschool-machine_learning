#!/usr/bin/env python3
"""Identity Block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in
    'Deep Residual Learning for Image Recognition (2015)'
    """
    conv_1 = K.layers.Conv2D(filters=filters[0],
                             kernel_size=(1, 1),
                             activation='relu',
                             padding='same',
                             kernel_initializer='he_normal')(A_prev)
    batch_norm_1 = K.layers.BatchNormalization()(conv_1)
    relu_1 = K.layers.Activation(activation='relu')(batch_norm_1)

    conv_2 = K.layers.Conv2D(filters=filters[1],
                             kernel_size=(3, 3),
                             activation='relu',
                             padding='same',
                             kernel_initializer='he_normal')(relu_1)
    batch_norm_2 = K.layers.BatchNormalization()(conv_2)
    relu_2 = K.layers.Activation(activation='relu')(batch_norm_2)

    conv_3 = K.layers.Conv2D(filters=filters[2],
                             kernel_size=(1, 1),
                             activation='relu',
                             padding='same',
                             kernel_initializer='he_normal')(relu_2)
    batch_norm_3 = K.layers.BatchNormalization()(conv_3)

    add = K.layers.Add()([batch_norm_3, A_prev])
    relu_3 = K.layers.Activation(activation='relu')(add)

    return relu_3
