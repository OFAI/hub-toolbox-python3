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
from scipy.spatial.distance import cdist, pdist, squareform
try: # for scikit-learn >= 0.18
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError: # lower scikit-learn versions
    from sklearn.cross_validation import StratifiedShuffleSplit
from hub_toolbox.IO import _check_vector_matrix_shape_fits_labels

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

def lp_norm(X:np.ndarray, p:float):
    """Calculate Minkowski distances between all pairs of vectors in `X`.

    Parameters
    ----------
    X : ndarray
        Input vector data

    p : float
        Minkowski norm

    Returns
    -------
    D : ndarray
        Distance matrix based on Lp-norm
    """
    return squareform(pdist(X, 'minkowski', p))

def sample_distance(X, y, sample_size, metric='euclidean', strategy='a'):
    """Calculate incomplete distance matrix.
    
    Parameters
    ----------
    X : ndarray
        Input vector data.

    y : ndarray
        Input labels (used for stratified sampling).

    sample_size : int or float
        If float, must be between 0.0 and 1.0 and represent the proportion of
        the dataset for which distances should be calculated to.
        If int, represents the absolute number of sample distances.
        NOTE: See also the notes to the return value `y_sample`!

    metric : any scipy.spatial.distance.cdist metric (default: 'euclidean')
        Metric used to calculate distances.

    strategy : 'a', 'b' (default: 'a')
        
        - 'a': Stratified sampling, for all points the distances to the
                same points are chosen.
        - 'b': Stratified sampling, for each point it is chosen independently,
                to which other points distances are calculated.
                NOTE: currently not implemented.

    Returns
    -------
    D : ndarray
        The ``n x s`` distance matrix, where ``n`` is the dataset size and
        ``s`` is the sample size.

    y_sample : ndarray
        The index array that determines, which column in `D` corresponds
        to which data point.
        
        NOTE: The size of `y_sample` may be slightly higher than defined by
        `sample_size` in order to meet stratification requirements!
        Thus, please always check the size in the downstream workflow.

    Notes
    -----
    Only calculate distances to a fixed number/fraction of all ``n`` points.
    These ``s`` points are sampled according to the chosen strategy (see above).
    In other words, calculate the distance from all points to each point
    in the sample to obtain a ``n x s`` distance matrix.
    
    """
    _check_vector_matrix_shape_fits_labels(X, y)
    n = X.shape[0]
    if not isinstance(sample_size, int):
        sample_size = int(sample_size * n)
    if strategy == 'a':
        try: # scikit-learn >= 0.18
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size)
            _, y_sample = sss.split(X=y, y=y)
        except TypeError: # scikit-learn < 0.18
            sss = StratifiedShuffleSplit(y=y, n_iter=1, test_size=sample_size)
            _, y_sample = next(iter(sss))
    elif strategy == 'b':
        raise NotImplementedError("Strategy 'b' is not yet implemented.")
        #=======================================================================
        # y_sample = np.zeros((n, sample_size))
        # try: # scikit-learn >= 0.18
        #     for i in range(n):
        #         sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size)
        #         _, y_sample[i, :] = sss.split(X=y, y=y)
        # except TypeError: # scikit-learn < 0.18
        #     for i in range(n):
        #         sss = StratifiedShuffleSplit(y=y, n_iter=1, test_size=sample_size)
        #         _, y_sample[i, :] = next(iter(sss))
        # # TODO will need to adapt cdist call below...
        #=======================================================================
    else:
        raise NotImplementedError("Strategy", strategy, "unknown.")
    
    D = cdist(X, X[y_sample, :], metric=metric)
    return D, y_sample

class Distance(Enum):
    """Enum for distance metrics.

    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              All functions now take str parameters directly.
    """
    cosine = 'cosine'
    euclidean = 'euclidean'
    skl = 'skl'
