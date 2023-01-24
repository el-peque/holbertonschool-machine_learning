#!/usr/bin/env python3
"""Input"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    regularizer = K.regularizers.L2(lambtha)
    L = len(layers)
    inputs = K.layers.Input(shape=(nx,))
    for i in range(L):
        if i == 0:
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=regularizer)(inputs)
        else:
            dropout = K.layers.Dropout(1 - keep_prob)(output)
            output = K.layers.Dense(layers[i],
                                    activation=activations[i],
                                    kernel_regularizer=regularizer)(dropout)
    model = K.models.Model(inputs=inputs, outputs=output)
    return model
