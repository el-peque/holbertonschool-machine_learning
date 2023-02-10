#!/usr/bin/env python3
"""Transition Layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    'Densely Connected Convolutional Networks'
    """
    batch_norm = K.layers.BatchNormalization()(X)
    nb_filters = int(nb_filters*compression)
    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(1, 1),
                           kernel_initializer='he_normal')(batch_norm)
    avgpool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(conv)
    return avgpool, nb_filters
