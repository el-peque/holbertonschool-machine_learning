#!/usr/bin/env python3
"""Inception Network"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds the ResNet-50 architecture as described in
    'Deep Residual Learning for Image Recognition (2015)'
    """
    inputs = K.Input((224, 224, 3))
    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=2,
                             padding='same',
                             kernel_initializer='he_normal')(inputs)
    batch_norm_1 = K.layers.BatchNormalization()(conv_1)
    relu_1 = K.layers.Activation(activation='relu')(batch_norm_1)
    maxpool_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2,
                                      padding='same')(relu_1)
    projection_1 = projection_block(maxpool_1, [64, 64, 256], s=1)
    identity_1 = identity_block(projection_1, [64, 64, 256])
    identity_2 = identity_block(identity_1, [64, 64, 256])
    projection_2 = projection_block(identity_2, [128, 128, 512], s=2)
    identity_3 = identity_block(projection_2, [128, 128, 512])
    identity_4 = identity_block(identity_3, [128, 128, 512])
    identity_5 = identity_block(identity_4, [128, 128, 512])
    projection_3 = projection_block(identity_5, [256, 256, 1024], s=2)
    identity_6 = identity_block(projection_3, [256, 256, 1024])
    identity_7 = identity_block(identity_6, [256, 256, 1024])
    identity_8 = identity_block(identity_7, [256, 256, 1024])
    identity_9 = identity_block(identity_8, [256, 256, 1024])
    identity_10 = identity_block(identity_9, [256, 256, 1024])
    projection_4 = projection_block(identity_10, [512, 512, 2048], s=2)
    identity_11 = identity_block(projection_4, [512, 512, 2048])
    identity_12 = identity_block(identity_11, [512, 512, 2048])
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1),
                                        padding='valid')(identity_12)
    softmax = K.layers.Dense(units=(1000),
                             activation='softmax',
                             kernel_initializer='he_normal')(avgpool)

    model = K.Model(inputs=inputs, outputs=softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
