#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """Trains a model using mini-batch gradient descent and early stopping"""
    callback = []
    if validation_data and learning_rate_decay:
        def lr_scheduler(epoch):
            """Learning rate scheduler"""
            return alpha / (1 + decay_rate * epoch)

        callback.append(K.callbacks.LearningRateScheduler(lr_scheduler,
                                                          verbose=1))

    if validation_data and early_stopping:
        callback.append(K.callbacks.EarlyStopping(monitor="val_loss",
                                                  mode="min",
                                                  patience=patience))

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          callbacks=callback,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
