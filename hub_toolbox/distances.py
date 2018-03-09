#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2018, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import ctypes
from multiprocessing import Pool, cpu_count, RawArray
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
try: # for scikit-learn >= 0.18
    from sklearn.model_selection import StratifiedShuffleSplit
except ImportError: # lower scikit-learn versions
    from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics.pairwise import pairwise_distances
from hub_toolbox.io import check_vector_matrix_shape_fits_labels
from hub_toolbox.logging import ConsoleLogging

__all__ = ['cosine_distance', 'euclidean_distance', 
           'lp_norm', 'sample_distance']

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
    """Calculate the euclidean distances between all pairs of vectors in `X`.
    
    Consider using sklearn.metric.pairwise.euclidean_distances for faster,
    but less accurate distances (not necessarily symmetric, too)."""
    return squareform(pdist(X, 'euclidean'))

def lp_norm(X:np.ndarray, Y:np.ndarray=None, p:float=None, n_jobs:int=1):
    """Calculate Minkowski distances with L^p norm.
    
    Calculate distances between all pairs of vectors within `X`, if `Y` is None.
    Otherwise calculate distances distances between all vectors in `X` against
    all vectors in `Y`. For example, this is useful if only distances from
    test data to training data are required.

    Parameters
    ----------
    X : ndarray
        Vector data (e.g. test set)

    Y : ndarray, optional, default: None
        Vector data (e.g. training set)

    p : float, default: None
        Minkowski norm

    n_jobs : int, default: 1
        Parallel computation with multiple processes. See the scikit-learn
        docs for for more details.

    Returns
    -------
    D : ndarray
        Distance matrix based on Lp-norm

    See also
    --------
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html
    """
    if p is None:
        raise ValueError("Please define the `p` parameter for lp_norm().")
    elif p == 1.: # Use efficient version for cityblock distances
        return pairwise_distances(X=X, Y=Y, metric='l1',
                                  n_jobs=n_jobs)
    elif p == 2.: # Use efficient version for Euclidean distances
        return pairwise_distances(X=X, Y=Y, metric='l2',
                                  n_jobs=n_jobs)
    else: # Use general, less efficient version for general Minkowski distances
        return pairwise_distances(X=X, Y=Y, metric='minkowski',
                                  n_jobs=n_jobs, **{'p' : p})

#===============================================================================
# #=============================================================================
# # 
# #                        m_p dissimilarity
# # 
# #=============================================================================
#===============================================================================
def _mp_load_shared_Y(Y_, n_bins_):
    global Y, n_bins
    Y = Y_
    n_bins = n_bins_

def _mp_load_shared_data(X_, Y_, p_, n_bins_, R_bins_, R_bins_np_,
                         X_bins_, X_bins_np_, Y_bins_, Y_bins_np_, mp_, mp_np_):
    global X, Y, n_bins, n_x, n_y, d, p
    global X_bins, X_bins_np, Y_bins, Y_bins_np, R_bins, R_bins_np, mp, mp_np
    X = X_
    Y = Y_
    n_bins = n_bins_
    n_x, d = X.shape
    n_y = Y.shape[0]
    p = p_
    R_bins = R_bins_
    R_bins_np = R_bins_np_
    X_bins = X_bins_
    X_bins_np = X_bins_np_
    Y_bins = Y_bins_
    Y_bins_np = Y_bins_np_
    mp = mp_
    mp_np = mp_np_

def _mp_find_bin_edges(i):
    return np.partition(Y[:, i], kth=kth)[kth]

def _mp_calc_histograms(i):
    bins = _mp_find_bin_edges(i)
    return np.histogram(Y[:, i], bins=bins)

def _mp_calc_histograms_n_bins(i):
    return np.histogram(Y[:, i], bins=n_bins)

def _mp_create_r_bins(i):
    hist, _ = histograms[i]
    for b in range(n_bins):
        R_bins_np[i, b, b:] = np.cumsum(hist[b:])
    R_bins_np[i] += np.triu(R_bins_np[i], k=1).T
    return

def _mp_estimate_r(i):
    # Binning. Values outside the range are binned into the first/last bin
    _, bin_edges = histograms[i]
    bin_x = np.digitize(X[:, i], bins=bin_edges)
    bin_x -= 1
    np.clip(bin_x, 0, n_bins-1, out=bin_x)
    bin_y = np.digitize(Y[:, i], bins=bin_edges)
    bin_y -= 1
    np.clip(bin_y, 0, n_bins-1, out=bin_y)
    X_bins_np[i, :] = bin_x
    Y_bins_np[i, :] = bin_y
    return

def _mp_calc_mp_dissim(x):
    mp_xy = np.zeros(n_y, dtype=float)
    for i in range(d):
        tmp = R_bins_np[i, X_bins_np[i, x], Y_bins_np[i, :]] / (n_x + n_y)
        tmp **= p
        mp_xy += tmp
    mp_xy /= d
    mp_xy **= (1. / p)
    mp_np[x, :] = mp_xy
    return

def mp_dissim(X:np.ndarray, Y:np.ndarray=None, p:float=2,
              n_bins:int=0, bin_size:str='range', n_jobs:int=1, verbose:int=0):
    """ Calculate m_p dissimilarity.

    The data-dependent m_p dissimilarity measure considers the relative
    positions of objects x and y with respect to the rest of the data
    distribution in each dimension [1]_.

    Parameters
    ----------
    X : ndarray
        Vector data (e.g. test set), shape (n_x, d)

    Y : ndarray, optional, default: None
        Vector data (e.g. training set), shape (n_y, d).
        Number of features ``d`` must be equal in `X` and `Y`.

    p : float, optional, default: 2
        Parameter, similar to `p` in Minkowski norm

    n_bins : int, optional, default: 0
        Number of bins for probability mass estimation

    bin_size : str, optional, default: 'range'
        Strategy for binning. May be one of:
            'range' ... create bins with uniform range length
            'mass'  ... create bins with approx. uniform mass

    n_jobs : int, optional, default: 1
        Parallel computation with multiple processes.

    verbose : int, optional, default: 0
        Increasing level of output

    Returns
    -------
    D : ndarray, shape (X.shape[0], Y.shape[0])
        m_p dissimilarity matrix

    References
    ----------
    .. [1] Aryal et al. (2017). Data-dependent dissimilarity measure: an
           effective alternative to geometric distance measures.
           Knowledge and Information Systems, Springer-Verlag London.
    """
    # Some preparation
    n_x, d = X.shape
    # All-against-all in X, or X against Y?
    if Y is None:
        Y = X
    n_y, d_y = Y.shape
    # X and Y must have same dimensionality
    assert d == d_y
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_bins = int(n_bins)
    if p == 0:
        log = ConsoleLogging()
        log.warning('Got mpDisSim parameter p=0. Changed to default '
                    'value p=2 instead, in order to avoid zero division.')
        p = 2.

    # RawArrays have no locks. Must take EXTREME CARE!!
    R_bins = RawArray(ctypes.c_int32, d * n_bins * n_bins)
    R_bins_np = np.frombuffer(R_bins, dtype=np.int32).reshape((d, n_bins, n_bins))
    X_bins = RawArray(ctypes.c_int32, d * n_x)
    X_bins_np = np.frombuffer(X_bins, dtype=np.int32).reshape((d, n_x))
    Y_bins = RawArray(ctypes.c_int32, d * n_y)
    Y_bins_np = np.frombuffer(Y_bins, dtype=np.int32).reshape((d, n_y))
    mp = RawArray(ctypes.c_double, n_x * n_y)
    mp_np = np.frombuffer(mp).reshape((n_x, n_y))

    global histograms, kth
    kth = np.arange(0, n_y)[0:n_y:int(n_y/n_bins)]
    if kth[-1] != n_y - 1:
        kth = np.append(kth, n_y-1)
    if verbose:
        print("Creating bins for estimating probability data mass.")
    with Pool(processes=n_jobs,
              initializer=_mp_load_shared_Y,
              initargs=(Y, n_bins)) as pool:
        if 'mass'.startswith(bin_size):
            histograms = pool.map(func=_mp_calc_histograms,
                                  iterable=range(d))
        elif 'range'.startswith(bin_size):
            histograms = pool.map(func=_mp_calc_histograms_n_bins,
                                  iterable=range(d))
        else:
            raise ValueError("{}' is not a valid value for `bin_size`. "
                             "Please use 'range' or 'mass'.".format(bin_size))
    # The second pool needs `histograms`
    with Pool(processes=n_jobs,
              initializer=_mp_load_shared_data,
              initargs=(X, Y, p, n_bins, R_bins, R_bins_np, X_bins, X_bins_np,
                        Y_bins, Y_bins_np, mp, mp_np)) as pool:
        pool.map(func=_mp_create_r_bins, iterable=range(d))
        if verbose:
            print("Estimating probability data mass in all regions R_i(x,y).")
        pool.map(func=_mp_estimate_r, iterable=range(d))
        if verbose:
            print("Calculating m_p dissimilarity for all pairs x, y.")
        pool.map(func=_mp_calc_mp_dissim, iterable=range(n_x))
    if verbose:
        print("Done.")
    return mp_np


def sample_distance(X, y, sample_size, metric='euclidean', strategy='a',
                    random_state=None):
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

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

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
    check_vector_matrix_shape_fits_labels(X, y)
    n = X.shape[0]
    if not isinstance(sample_size, int):
        sample_size = int(sample_size * n)
    if strategy == 'a':
        try: # scikit-learn == 0.18
            sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size,
                                         random_state=random_state)
            _, y_sample = sss.split(X=X, y=y)
        except ValueError: # scikit-learn >= 0.18.1
            _, y_sample = next(sss.split(X=X, y=y))
        except TypeError: # scikit-learn < 0.18
            sss = StratifiedShuffleSplit(y=y, n_iter=1, test_size=sample_size,
                                         random_state=random_state)
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
