#!/usr/bin/env python3
"""Sensitivity"""
import numpy as np


def sensitivity(confusion):
    """Calculates the sensitivity for each class in a confusion matrix"""
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        tp = confusion[i, i]
        fn = np.sum(confusion[i, :]) - tp
        sensitivity[i] = tp / (fn + tp)

    return sensitivity
