#!/usr/bin/env python3
"""Predict"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    return network.predict(data, verbose=verbose)
