#!/usr/bin/env python3
"""Dense Block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in
    'Densely Connected Convolutional Networks'
    """
    for i in range(layers):
        batch_norm = K.layers.BatchNormalization()(X)
        relu = K.layers.Activation(activation='relu')(batch_norm)
        bottleneck = K.layers.Conv2D(filters=growth_rate*4,
                                     kernel_size=(1, 1),
                                     padding='same',
                                     kernel_initializer='he_normal')(relu)
        batch_norm_1 = K.layers.BatchNormalization()(bottleneck)
        relu_1 = K.layers.Activation(activation='relu')(batch_norm_1)
        conv = K.layers.Conv2D(filters=growth_rate,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer='he_normal')(relu_1)
        concat = K.layers.concatenate([conv, X])
        nb_filters += growth_rate
        X = concat

    return concat, nb_filters
