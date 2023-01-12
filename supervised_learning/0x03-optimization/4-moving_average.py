#!/usr/bin/env python3
"""Moving Average"""
import numpy as np


def moving_average(data, beta):
    """Calculates the weighted moving average of a data set"""
    v = 0
    avgs = []
    for i in range(len(data)):
        v = beta * v + (1 - beta) * data[i]
        avgs.append(v / (1 - (beta ** (i + 1))))
    return avgs
