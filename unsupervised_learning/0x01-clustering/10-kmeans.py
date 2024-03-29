#!/usr/bin/env python3
"""K-means sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset"""
    kmeans = sklearn.cluster.KMeans(k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
