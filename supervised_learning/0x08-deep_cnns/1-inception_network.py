#!/usr/bin/env python3
"""Inception Network"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Builds the Googlenet inception network"""
    # Configuration dicts
    conv_1 = {'filters': 64, 'kernel_size': (7, 7),
              'strides': (2, 2), 'activation': 'relu',
              'padding': 'same', 'kernel_initializer': 'he_normal'}
    maxpool_1 = {'pool_size': (3, 3), 'strides': (2, 2), 'padding': 'same'}
    conv_2 = {'filters': 64, 'kernel_size': (1, 1),
              'strides': (1, 1), 'activation': 'relu',
              'padding': 'valid', 'kernel_initializer': 'he_normal'}
    conv_3 = {'filters': 192, 'kernel_size': (3, 3),
              'strides': (1, 1), 'activation': 'relu',
              'padding': 'same', 'kernel_initializer': 'he_normal'}
    conv_4 = {'filters': 64, 'kernel_size': (1, 1),
              'strides': (1, 1), 'activation': 'relu',
              'padding': 'same', 'kernel_initializer': 'he_normal'}
    avgpool_1 = {'pool_size': (7, 7), 'strides': (1, 1), 'padding': 'valid'}
    fc_1 = {'units': (1000), 'activation': 'softmax',
            'kernel_initializer': 'he_normal'}

    inputs = K.Input(shape=(224, 224, 3))
    lay_1 = K.layers.Conv2D(**conv_1)(inputs)
    lay_2 = K.layers.MaxPooling2D(**maxpool_1)(lay_1)
    lay_3 = K.layers.Conv2D(**conv_2)(lay_2)
    lay_4 = K.layers.Conv2D(**conv_3)(lay_3)
    lay_5 = K.layers.MaxPooling2D(**maxpool_1)(lay_4)
    lay_6 = inception_block(lay_5, (64, 96, 128, 16, 32, 32))
    lay_7 = inception_block(lay_6, (128, 128, 192, 32, 96, 64))
    lay_8 = K.layers.MaxPooling2D(**maxpool_1)(lay_7)
    lay_9 = inception_block(lay_8, (192, 96, 208, 16, 48, 64))
    lay_10 = inception_block(lay_9, (160, 112, 224, 24, 64, 64))
    lay_11 = inception_block(lay_10, (128, 128, 256, 24, 64, 64))
    lay_12 = inception_block(lay_11, (112, 144, 288, 32, 64, 64))
    lay_13 = inception_block(lay_12, (256, 160, 320, 32, 128, 128))
    lay_14 = K.layers.MaxPooling2D(**maxpool_1)(lay_13)
    lay_15 = inception_block(lay_14, (256, 160, 320, 32, 128, 128))
    lay_16 = inception_block(lay_15, (384, 192, 384, 48, 128, 128))
    lay_17 = K.layers.AveragePooling2D(**avgpool_1)(lay_16)
    lay_18 = K.layers.Dropout(rate=(0.4))(lay_17)
    lay_19 = K.layers.Dense(**fc_1)(lay_18)
    model = K.Model(inputs=inputs, outputs=lay_19)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
