#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
"""Train"""


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier"""
