#!/usr/bin/env python3
"""Sequential"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)
    for i, (layer, activation) in enumerate(zip(layers, activations)):
        model.add(K.layers.Dense(layer,
                                 activation=activation,
                                 kernel_regularizer=regularizer,
                                 input_shape=(nx,)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
