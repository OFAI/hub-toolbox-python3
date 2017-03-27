#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2017, Dominik Schnitzer and Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import multiprocessing as mp
import numpy as np
from scipy import stats
from scipy.sparse.base import issparse
from hub_toolbox import IO, Logging

__all__ = ['hubness']

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
    log = Logging.ConsoleLogging()
    IO.check_is_nD_array(arr=D, n=2, arr_type='Distance')
    IO.check_valid_metric_parameter(metric)
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
    tasks = []

    batches = []
    batch_size = n // NUMBER_OF_PROCESSES
    for i in range(NUMBER_OF_PROCESSES-1):
        batches.append(np.arange(i*batch_size, (i+1)*batch_size))
    batches.append(np.arange((NUMBER_OF_PROCESSES-1)*batch_size, n))

    for idx, batch in enumerate(batches):
        submatrix = D[batch[0]:batch[-1]+1]
        tasks.append((_partial_hubness,
                      (k, kth, d_self, log, sort_order,
                      batch, submatrix, idx, n, m, verbose)))

    task_queue = mp.Queue()
    done_queue = mp.Queue()

    for task in tasks:
        task_queue.put(task)

    for i in range(NUMBER_OF_PROCESSES):
        mp.Process(target=_worker, args=(task_queue, done_queue)).start()

    for i in range(len(tasks)):
        rows, Dk_part = done_queue.get()
        #D_k[:, rows[0]:rows[-1]+1] = Dk_part
        D_k[rows[0]:rows[-1]+1, :] = Dk_part

    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

    # k-occurence
    N_k = np.bincount(D_k.astype(int).ravel(), minlength=m)
    # Hubness
    S_k = stats.skew(N_k)

    if verbose:
        log.message("Hubness calculation done.", flush=True)

    # return hubness, k-nearest neighbors, N occurence
    return S_k, D_k.T, N_k

def _worker(work_input, work_output):
    """A helper function for cv parallelization."""
    for func, args in iter(work_input.get, 'STOP'):
        result = _calculate(func, args)
        work_output.put(result)

def _calculate(func, args):
    """A helper function for cv parallelization."""
    return func(*args)

def _partial_hubness(k, kth, d_self, log, sort_order,
                     rows, submatrix, idx, n, m, verbose):
    """Parallel hubness calculation: Get k nearest neighbors for all points
    in 'rows'"""
    #===========================================================================
    # if sort_order == 1:
    #     kth = np.arange(k)
    # elif sort_order == -1:
    #     kth = np.arange(m - k, m)
    #===========================================================================

    Dk = np.zeros((len(rows), k), dtype=np.float64)

    for i, row in enumerate(submatrix):
        if verbose and ((rows[i]+1)%10000==0 or rows[i]+1==n):
            log.message("NN: {} of {}.".format(rows[i]+1, n), flush=True)
        if issparse(submatrix):
            d = row.toarray().ravel() # dense copy of one row
        else: # normal ndarray
            d = row
        d[rows[i]] = d_self
        d[~np.isfinite(d)] = d_self
        # randomize the distance matrix rows to avoid the problem case
        # if all numbers to sort are the same, which would yield high
        # hubness, even if there is none
        rp = np.random.permutation(m)
        d2 = d[rp]
        #d2idx = np.argsort(d2, axis=0)[::sort_order]
        #Dk[i, :] = rp[d2idx[:k]]
        d2idx = np.argpartition(d2, kth=kth)[::sort_order]
        Dk[i, :] = rp[d2idx[:k]]

    return [rows, Dk]

def _hubness_no_multiprocessing(D:np.ndarray, k:int=5, metric='distance',
                                verbose:int=0, random_state=None):
    """ Hubness calculations without multiprocessing overhead. """
    log = Logging.ConsoleLogging()
    IO.check_is_nD_array(arr=D, n=2, arr_type='Distance')
    IO.check_valid_metric_parameter(metric)
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

if __name__ == '__main__':
    # Simple test case
    from hub_toolbox.IO import load_dexter
    dexter_distance, l, v = load_dexter()
    Sn, Dk, Nk = hubness(dexter_distance)
    print("Hubness =", Sn)
