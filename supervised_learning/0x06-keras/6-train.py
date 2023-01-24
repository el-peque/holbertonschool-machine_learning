#!/usr/bin/env python3
"""Train and Validate"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and early stopping"""
    callback = None
    if validation_data and early_stopping:
        callback = K.callbacks.EarlyStopping(monitor="val_loss",
                                             patience=patience)
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          callbacks=[callback],
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
