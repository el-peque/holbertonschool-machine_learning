#!/usr/bin/env python3
"""Precision"""
import numpy as np


def precision(confusion):
    """Calculates the precision for each class in a confusion matrix"""
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    for i in range(classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        precision[i] = tp / (tp + fp)

    return precision
