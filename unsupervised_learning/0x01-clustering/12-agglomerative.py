#!/usr/bin/env python3
"""Agglomerative"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset"""
    clusters = scipy.cluster.hierarchy.linkage(X, method="ward")
    dn = scipy.cluster.hierarchy.dendrogram(clusters, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(clusters, dist, 'distance')
    plt.show()
    return clss
