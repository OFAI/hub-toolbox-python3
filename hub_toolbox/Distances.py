#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2016, Dominik Schnitzer and Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

from enum import Enum
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cross_validation import StratifiedKFold

def cosine_distance(X):
    """Calculate the cosine distance between all pairs of vectors in `X`."""
    xn = np.sqrt(np.sum(X**2, 1))
    Y = X / xn[:, np.newaxis]
    del xn
    D = 1. - Y.dot(Y.T)
    del Y
    D[D < 0] = 0
    D = np.triu(D, 1) + np.triu(D, 1).T
    return D

def euclidean_distance(X):
    """Calculate the euclidean distances between all pairs of vectors in `X`."""
    return squareform(pdist(X, 'euclidean'))

def sample_distance(X, y, metric, sample_size, strategy):
    """Calculate incomplete distance matrix.
    
    Only calculate distances to a fixed number/fraction of all other points.
    Atm, this means that still all distances are calculated and non-sample
    points are simply masked by `np.nan`. Efficient version will be implemented
    after evaluation shows at least reasonable performance for kNN and MP.
    
    Parameters
    ----------
    X : ndarray
        Input vector data.
    y : ndarray
        Input labels (used for stratified sampling).
    metric : 'cosine' or metric understood by scipy.spatial.distance.pdist
        Metric used to calculate distances.
    sample_size : int or float
        If float, must be between 0.0 and 1.0 and represent the proportion of
        the dataset for which distances should be calculated per point.
        If int, represents the absolute number of sample distances.
    strategy : 'a', 'b'
        
        - 'a': Stratified sampling, for all points the distances to the
                same points are chosen.
        - 'b': Stratified sampling, for each point it is chosen independently,
                to which other points distances are calculated.

    Returns
    -------
    D : ndarray
        Distance matrix. Non-sample distances are masked with `np.nan`. This
        is subject to change upon successful evaluation: Only sample distances
        will be calculated and saved in a more efficient data structure (most
        likely some sparse matrix, like LIL).
    """
    if metric == 'cosine':
        D = cosine_distance(X)
    elif metric == 'euclidean':
        D = euclidean_distance(X)
    else:
        D = squareform(pdist(X, metric=metric))

    n = D.shape[0]
    if not isinstance(sample_size, int):
        sample_size = int(sample_size * n)

    if strategy == 'a':
        y_nonsample, _ = next(iter(
            StratifiedKFold(y, n_folds=n//sample_size, shuffle=True)))
        D[:, y_nonsample] = np.nan
    elif strategy == 'b':
        for i in range(n):
            y_nonsample, _ = next(iter(
                StratifiedKFold(y, n_folds=n//sample_size, shuffle=True)))
            D[i, y_nonsample] = np.nan
    else:
        raise NotImplementedError("Strategy", strategy, "unknown.")

    return D

class Distance(Enum):
    """Enum for distance metrics.

    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              All functions now take str parameters directly.
    """
    cosine = 'cosine'
    euclidean = 'euclidean'
    skl = 'skl'
