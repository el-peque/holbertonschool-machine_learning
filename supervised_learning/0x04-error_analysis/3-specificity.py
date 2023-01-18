#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """Calculates the specificity for each class in a confusion matrix"""
    classes = confusion.shape[0]
    specificity = np.zeros(classes)
    conf = confusion.T
    for i in range(classes):
        tp = conf[i, i]
        tn = np.sum(np.sum(conf[:i, :i]) + np.sum(conf[:i, i+1:]) +
                    np.sum(conf[i+1:, :i]) + np.sum(conf[i+1:, i+1:]))
        fp = np.sum(conf[i, :]) - tp
        specificity[i] = tn / (tn + fp)

    return specificity
