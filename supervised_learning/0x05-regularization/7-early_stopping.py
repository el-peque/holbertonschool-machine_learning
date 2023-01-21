#!/usr/bin/env python3
"""Early Stopping"""
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Determines if you should stop gradient descent early"""
    res = False
    if (opt_cost - cost) > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        res = True

    return (res, count)
