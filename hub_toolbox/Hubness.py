#!/usr/bin/env python3
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

import numpy as np
from scipy import stats
from scipy.sparse.base import issparse
from hub_toolbox import IO, Logging
import sys

def hubness(D:np.ndarray, k:int=5, metric='distance', verbose:int=0):
    """Compute hubness of a distance matrix.

    Hubness [1]_ is the skewness of the `k`-occurrence histogram (reverse
    nearest neighbor count, i.e. how often does a point occur in the
    `k`-nearest neighbor lists of other points).

    Parameters
    ----------
    D : ndarray
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 5)
        Neighborhood size for `k`-occurence.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    Returns
    -------
    S_k : float
        Hubness (skewness of `k`-occurence distribution)
    D_k : ndarray
        `k`-nearest neighbor lists
    N_k : ndarray
        `k`-occurence list

    References
    ----------
    .. [1] Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010).
           Hubs in Space : Popular Nearest Neighbors in High-Dimensional Data.
           Journal of Machine Learning Research, 11, 2487–2531. Retrieved from
           http://jmlr.csail.mit.edu/papers/volume11/radovanovic10a/
           radovanovic10a.pdf
    """
    log = Logging.ConsoleLogging()
    IO._check_distance_matrix_shape(D)
    IO._check_valid_metric_parameter(metric)
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
    if metric == 'similarity':
        d_self = -np.inf
        sort_order = -1

    if verbose:
        log.message("Hubness calculation (skewness of {}-occurence)".format(k))
    D = D.copy()
    D_k = np.zeros((k, D.shape[1]), dtype=np.float32)
    n = D.shape[0]

    if issparse(D):
        pass # correct self-distance must be ensured upstream for sparse
    else:
        # Set self dist to inf
        np.fill_diagonal(D, d_self)
        # make non-finite (NaN, Inf) appear at the end of the sorted list
        D[~np.isfinite(D)] = d_self

    for i in range(n):
        if verbose and ((i+1)%10000==0 or i+1==n):
            log.message("NN: {} of {}.".format(i+1, n), flush=True)
        if issparse(D):
            d = D[i, :].toarray().ravel() # dense copy of one row
        else: # normal ndarray
            d = D[i, :]
        d[i] = d_self
        d[~np.isfinite(d)] = d_self
        # Randomize equal values in the distance matrix rows to avoid the 
        # problem case if all numbers to sort are the same, which would yield 
        # high hubness, even if there is none.
        rp = np.random.permutation(n)
        d2 = d[rp]
        d2idx = np.argsort(d2, axis=0)[::sort_order]
        D_k[:, i] = rp[d2idx[0:k]]

    # N-occurence
    N_k = np.bincount(D_k.astype(int).ravel(), minlength=n)
    # Hubness
    S_k = stats.skew(N_k)

    # return k-hubness, k-nearest neighbors, k-occurence
    if verbose:
        log.message("Hubness calculation done.", flush=True)
    return S_k, D_k, N_k


class Hubness(): # pragma: no cover
    """
    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              Please use static functions instead.
    """

    def __init__(self, D, k:int=5, isSimilarityMatrix:bool=False):
        self.log = Logging.ConsoleLogging()
        if isinstance(D, np.memmap):
            self.D = D
        else:
            self.D = IO.copy_D_or_load_memmap(D, writeable=False)
        self.k = k
        if isSimilarityMatrix:
            self.d_self = -np.inf
            self.sort_order = -1 # descending, interested in highest similarity
        else:
            self.d_self = np.inf
            self.sort_order = 1 # ascending, interested in smallest distance
        np.random.seed()

    def calculate_hubness(self, debug=False):
        """Calculate hubness.

        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        print("DEPRECATED: Please use Hubness.hubness().", file=sys.stderr)
        if self.sort_order == 1:
            metric = 'distance'
        elif self.sort_order == -1:
            metric = 'similarity'
        else:
            raise ValueError("sort_order must be -1 or 1.")
        return hubness(self.D, self.k, metric, debug)

if __name__ == '__main__':
    # Simple test case
    from hub_toolbox.HubnessAnalysis import load_dexter
    dexter_distance, l, v = load_dexter()
    Sn, Dk, Nk = hubness(dexter_distance)
    print("Hubness =", Sn)
