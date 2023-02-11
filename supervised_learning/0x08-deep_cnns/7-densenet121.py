#!/usr/bin/env python3
"""DenseNet-121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in
    'Densely Connected Convolutional Networks'
    """
    nb_filters = growth_rate * 2
    X = K.Input((224, 224, 3))
    batchnorm_1 = K.layers.BatchNormalization()(X)
    relu_1 = K.layers.Activation(activation='relu')(batchnorm_1)
    conv_1 = K.layers.Conv2D(filters=64,
                             kernel_size=(7, 7),
                             strides=2,
                             padding='same',
                             kernel_initializer='he_normal')(relu_1)
    maxpool_1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                      strides=2,
                                      padding='same')(conv_1)
    dense_1, nb_filters = dense_block(maxpool_1, nb_filters, growth_rate, 6)
    trans_1, nb_filters = transition_layer(dense_1, nb_filters, compression)
    dense_2, nb_filters = dense_block(trans_1, nb_filters, growth_rate, 12)
    trans_2, nb_filters = transition_layer(dense_2, nb_filters, compression)
    dense_3, nb_filters = dense_block(trans_2, nb_filters, growth_rate, 24)
    trans_3, nb_filters = transition_layer(dense_3, nb_filters, compression)
    dense_4, nb_filters = dense_block(trans_3, nb_filters, growth_rate, 16)
    avgpool = K.layers.AveragePooling2D(pool_size=(7, 7),
                                        strides=(1, 1),
                                        padding='valid')(dense_4)
    softmax = K.layers.Dense(units=(1000),
                             activation='softmax',
                             kernel_initializer='he_normal')(avgpool)

    model = K.Model(inputs=X, outputs=softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
