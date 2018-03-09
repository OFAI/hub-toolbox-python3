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
from functools import partial
import multiprocessing as mp
from multiprocessing import RawArray, Pool
import numpy as np
from scipy import stats
from scipy.sparse.base import issparse
from sklearn.metrics.pairwise import pairwise_distances
from hub_toolbox import io, logging

__all__ = ['hubness', 'hubness_from_vectors']


def _hubness_load_shared_data(D_, D_k_):
    global D, D_k
    D = D_
    D_k = D_k_
    return

def _hubness_nearest_neighbors(i, k, n, m, d_self, metric, 
                               kth, sort_order, log, verbose):
    if verbose and ((i+1)%10000==0 or i+1==n):
        log.message("NN: {} of {}.".format(i+1, n), flush=True)
    if issparse(D):
        d = D[i, :].toarray().ravel() # dense copy of one row
    else: # normal ndarray
        d = D[i, :]
    if n == m:
        d[i] = d_self
    else: # this does not hold for general dissimilarities
        if metric == 'distance':
            d[d==0] = d_self
    d[~np.isfinite(d)] = d_self
    # Randomize equal values in the distance matrix rows to avoid the
    # problem case if all numbers to sort are the same, which would yield
    # high hubness, even if there is none.
    rp = np.random.permutation(m)
    d2 = d[rp]
    d2idx = np.argpartition(d2, kth=kth)
    D_k[i, :] = rp[d2idx[kth]][::sort_order]
    return

def hubness(D:np.ndarray, k:int=5, metric='distance',
            verbose:int=0, n_jobs:int=1, random_state=None):
    """Compute hubness of a distance matrix.

    Hubness [1]_ is the skewness of the `k`-occurrence histogram (reverse
    nearest neighbor count, i.e. how often does a point occur in the
    `k`-nearest neighbor lists of other points).

    Parameters
    ----------
    D : ndarray
        The ``n x n`` symmetric distance (similarity) matrix or
        an ``n x m`` partial distances matrix (e.g. for train/test splits,
        with test objects in rows, train objects in column)
        
        NOTE: Partial distance matrices MUST NOT contain self distances.

    k : int, optional (default: 5)
        Neighborhood size for `k`-occurrence.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    n_jobs : int, optional (default: 1)
        Number of parallel processes spawned for hubness calculation.
        Value 1 (default): One process (not using multiprocessing)
        Value (-1): As many processes as number of available CPUs.

    random_state : int, optional
        Seed the RNG for reproducible results.
        
        NOTE: Currently only compatible with `n_jobs`=1

    Returns
    -------
    S_k : float
        Hubness (skewness of `k`-occurrence distribution)
    D_k : ndarray
        `k`-nearest neighbor lists
    N_k : ndarray
        `k`-occurrence list

    References
    ----------
    .. [1] Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010).
           Hubs in Space : Popular Nearest Neighbors in High-Dimensional Data.
           Journal of Machine Learning Research, 11, 2487–2531. Retrieved from
           http://jmlr.csail.mit.edu/papers/volume11/radovanovic10a/
           radovanovic10a.pdf
    """
    # Don't use multiprocessing environment when using only one job
    if n_jobs == 1:
        return _hubness_no_multiprocessing(D=D, k=k, metric=metric,
                                           verbose=verbose, random_state=random_state)
    if random_state is not None:
        raise ValueError("Seeding the RNG is not compatible with using n_jobs > 1.")
    log = logging.ConsoleLogging()
    io.check_is_nD_array(arr=D, n=2, arr_type='Distance')
    io.check_valid_metric_parameter(metric)
    n, m = D.shape
    if k >= m:
        k_old = k
        k = m - 1
        log.warning("Reducing k from {} to {}, so that it is less than "
                    "the total number of neighbors.".format(k_old, k))
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
        kth = np.arange(k)
    if metric == 'similarity':
        d_self = -np.inf
        sort_order = -1
        kth = np.arange(m - k, m)

    if verbose:
        log.message("Hubness calculation (skewness of {}-occurrence)".format(k))

    # Initialization
    D = D.copy()
    D_k = np.zeros((n, k), dtype=np.float64)

    if issparse(D):
        pass # correct self-distance must be ensured upstream for sparse
    else:
        if n == m:
            # Set self dist to inf
            np.fill_diagonal(D, d_self)
        else:
            pass # Partial distance matrices MUST NOT contain self distances
        # make non-finite (NaN, Inf) appear at the end of the sorted list
        D[~np.isfinite(D)] = d_self

    # Parallelization
    if n_jobs == -1: # take all cpus
        NUMBER_OF_PROCESSES = mp.cpu_count() # @UndefinedVariable
    else:
        NUMBER_OF_PROCESSES = n_jobs
    D_k_ctype = RawArray(ctypes.c_int32, n*k)
    D_k = np.frombuffer(D_k_ctype, dtype=np.int32).reshape((n, k))
    with Pool(processes=NUMBER_OF_PROCESSES,
              initializer=_hubness_load_shared_data,
              initargs=(D, D_k, )) as pool:
        for _ in pool.imap(
            func=partial(_hubness_nearest_neighbors, k=k, n=n, m=m, 
                         d_self=d_self, metric=metric, kth=kth, 
                         sort_order=sort_order, log=log, verbose=verbose),
            #chunksize=int(1e2),
            iterable=range(n)):
            pass # results handled within func

    # k-occurrence
    N_k = np.bincount(D_k.astype(int).ravel(), minlength=m)
    # Hubness
    S_k = stats.skew(N_k)

    if verbose:
        log.message("Hubness calculation done.", flush=True)

    # return hubness, k-nearest neighbors, N occurence
    return S_k, D_k.T, N_k

def _hubness_no_multiprocessing(D:np.ndarray, k:int=5, metric='distance',
                                verbose:int=0, random_state=None):
    """ Hubness calculations without multiprocessing overhead. """
    log = logging.ConsoleLogging()
    io.check_is_nD_array(arr=D, n=2, arr_type='Distance')
    io.check_valid_metric_parameter(metric)
    n, m = D.shape
    if k >= m:
        k_old = k
        k = m - 1
        log.warning("Reducing k from {} to {}, so that it is less than "
                    "the total number of neighbors.".format(k_old, k))
    if metric == 'distance':
        d_self = np.inf
        sort_order = 1
        kth = np.arange(k)
    if metric == 'similarity':
        d_self = -np.inf
        sort_order = -1
        kth = np.arange(n - k, n)

    if verbose:
        log.message("Hubness calculation (skewness of {}-occurence)".format(k))
    D = D.copy()
    D_k = np.zeros((n, k), dtype=np.float64)
    rnd = np.random.RandomState(random_state)

    if issparse(D):
        pass # correct self-distance must be ensured upstream for sparse
    else:
        if n == m:
            # Set self dist to inf
            np.fill_diagonal(D, d_self)
        else:
            pass # a partial distances matrix should not contain self distances
        # make non-finite (NaN, Inf) appear at the end of the sorted list
        D[~np.isfinite(D)] = d_self

    for i in range(n):
        if verbose and ((i+1)%10000==0 or i+1==n):
            log.message("NN: {} of {}.".format(i+1, n), flush=True)
        if issparse(D):
            d = D[i, :].toarray().ravel() # dense copy of one row
        else: # normal ndarray
            d = D[i, :]
        if n == m:
            d[i] = d_self
        else: # this does not hold for general dissimilarities
            if metric == 'distance':
                d[d==0] = d_self
        d[~np.isfinite(d)] = d_self
        # Randomize equal values in the distance matrix rows to avoid the
        # problem case if all numbers to sort are the same, which would yield
        # high hubness, even if there is none.
        rp = rnd.permutation(m)
        d2 = d[rp]
        d2idx = np.argpartition(d2, kth=kth)
        D_k[i, :] = rp[d2idx[kth]][::sort_order]

    # N-occurence
    N_k = np.bincount(D_k.astype(int).ravel(), minlength=m)
    # Hubness
    S_k = stats.skew(N_k)

    # return k-hubness, k-nearest neighbors, k-occurence
    if verbose:
        log.message("Hubness calculation done.", flush=True)
    return S_k, D_k.T, N_k

def hubness_from_vectors(X:np.ndarray, Y:np.ndarray=None, k:int=5,
                         metric='euclidean', verbose:int=0,
                         n_jobs:int=1, random_state=None):
    """Compute hubness from vectors.

    Hubness [1]_ is the skewness of the `k`-occurrence histogram (reverse
    nearest neighbor count, i.e. how often does a point occur in the
    `k`-nearest neighbor lists of other points).

    Parameters
    ----------
    X : ndarray, shape (n_test, n_features)
        Test vectors. Will compute distances between each vector in X
        and all vectors in Y.

    Y : ndarray, shape (n_train, n_features)
        Training vectors. If None, compute all-against-all distances in X.

    k : int, optional (default: 5)
        Neighborhood size for `k`-occurrence.

    metric : string, default
        The metric used for distance calculations. May be any of the
        allowed metrics from http://scikit-learn.org/stable/modules/
        generated/sklearn.metrics.pairwise.pairwise_distances.html

    verbose : int, optional (default: 0)
        Increasing level of output (progress report).

    n_jobs : int, optional (default: 1)
        Number of parallel processes spawned for hubness calculation.
        Value 1 (default): One process (not using multiprocessing)
        Value (-1): As many processes as number of available CPUs.

    random_state : int, optional
        Seed the RNG for reproducible results.
        
        NOTE: Currently only compatible with `n_jobs`=1

    Returns
    -------
    S_k : float
        Hubness (skewness of `k`-occurrence distribution)
    D_k : ndarray
        `k`-nearest neighbor lists
    N_k : ndarray
        `k`-occurrence list

    References
    ----------
    .. [1] Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010).
           Hubs in Space : Popular Nearest Neighbors in High-Dimensional Data.
           Journal of Machine Learning Research, 11, 2487–2531. Retrieved from
           http://jmlr.csail.mit.edu/papers/volume11/radovanovic10a/
           radovanovic10a.pdf
    """
    if Y is None:
        Y = X
        kth = np.arange(k + 1)
        start = 1
        end = k + 1
    else:
        kth = np.arange(k)
        start = 0
        end = k
    assert end - start == k, f'Implementation error'
    n_test, m_test = X.shape
    n_train, m_train = Y.shape
    assert m_test == m_train, f'Number of features do not match'
    Dk = np.empty((n_test, k), dtype=np.int32)
    for i in range(n_test):
        d = pairwise_distances(X[i, :].reshape(1, -1), Y, metric, n_jobs)
        nn = np.argpartition(d, kth=kth)[0, start:end]
        Dk[i, :] = nn
    # N-occurence
    Nk = np.bincount(Dk.astype(int).ravel(), minlength=n_train)
    # Hubness
    Sk = stats.skew(Nk)
    return Sk, Dk, Nk

if __name__ == '__main__':
    # Simple test case
    from hub_toolbox.io import load_dexter
    dexter_distance, l, v = load_dexter()
    Sn, Dk, Nk = hubness(dexter_distance)
    Snv, Dkv, Nkv = hubness_from_vectors(v, metric='cosine')
    print("Hubness =", Sn, Snv)
