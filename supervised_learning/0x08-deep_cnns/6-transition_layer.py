#!/usr/bin/env python3
"""Transition Layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in
    'Densely Connected Convolutional Networks'
    """
    nb_filters = int(nb_filters * compression)
    batch_norm = K.layers.BatchNormalization()(X)
    relu = K.layers.Activation('relu')(batch_norm)
    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(1, 1),
                           padding='same',
                           kernel_initializer='he_normal')(relu)
    avgpool = K.layers.AveragePooling2D(pool_size=2, strides=2)(conv)
    return avgpool, nb_filters
