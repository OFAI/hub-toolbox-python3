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
from multiprocessing import RawArray, Pool, cpu_count
import numpy as np
from scipy import stats
from scipy.sparse.base import issparse
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_random_state
from hub_toolbox import io
from hub_toolbox.htlogging import ConsoleLogging
from hub_toolbox.utils import SynchronizedCounter

__all__ = ['Hubness', 'hubness', 'hubness_from_vectors']
VALID_METRICS = ['euclidean', 'cosine', 'precomputed']

log = ConsoleLogging()

def _k_neighbors_initializer(X_=None, Y_=None, Dk_=None,
                             X_norm_=None, Y_norm_=None,
                             counter_=None):
    global X, Y, Dk, X_norm, Y_norm, counter
    X = X_
    Y = Y_
    Dk = Dk_
    X_norm = X_norm_
    Y_norm = Y_norm_
    counter = counter_
    return

def _k_neighbors_parallel(i, kth, start, end, metric,
                          verbose, batch_size, n_batches):
    progress = counter.increment_and_get_value()
    if verbose > 1 \
        or verbose > 0 and (progress % 10 == 0 or progress+1 == n_batches):
        log.message(f'k neighbors (multiprocessing): batch '
                    f'{progress+1}/{n_batches} with batch size '
                    f'{batch_size}.', flush=True)
    d = pairwise_distances(
        X[i*batch_size:(i+1)*batch_size, :], Y, metric, squared=True,
        X_norm_squared=X_norm[i*batch_size:(i+1)*batch_size].reshape(1, -1),
        Y_norm_squared=Y_norm)
    nn = np.argpartition(d, kth=kth)[:, start:end]
    Dk[i*batch_size:(i+1)*batch_size, :] = nn
    return


def _hubness_load_shared_data(D_, D_k_):
    global D, D_k
    D = D_
    D_k = D_k_
    return

def _hubness_nearest_neighbors(i, n, m, d_self, metric, 
                               kth, sort_order, log, verbose, shuffle_equal):
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
    if shuffle_equal:
        # Randomize equal values in the distance matrix rows to avoid the
        # problem case if all numbers to sort are the same, which would yield
        # high hubness, even if there is none.
        rp = np.random.permutation(m)
        d2 = d[rp]
        d2idx = np.argpartition(d2, kth=kth)
        D_k[i, :] = rp[d2idx[kth]][::sort_order]
    else:
        d_idx = np.argpartition(d, kth=kth)
        D_k[i, :] = d_idx[kth][::sort_order]
    return

def hubness(D:np.ndarray, k:int=5, metric='distance',
            verbose:int=0, n_jobs:int=1,
            random_state=None, shuffle_equal=True):
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

    shuffle_equal : bool, optional
        If true, shuffle neighbors with identical distances to avoid
        artifact hubness.
        NOTE: This is especially useful for secondary distance measures
        with a restricted number of possible values, e.g. SNN or MP empiric.

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
        return _hubness_no_multiprocessing(D=D,
                                           k=k,
                                           metric=metric,
                                           verbose=verbose,
                                           random_state=random_state,
                                           shuffle_equal=shuffle_equal)
    if random_state is not None:
        raise ValueError("Seeding the RNG is not compatible with using n_jobs > 1.")
    log = ConsoleLogging()
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
            func=partial(_hubness_nearest_neighbors, n=n, m=m, 
                         d_self=d_self, metric=metric, kth=kth, 
                         sort_order=sort_order, log=log, verbose=verbose,
                         shuffle_equal=shuffle_equal),
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
    return S_k, D_k, N_k

def _hubness_no_multiprocessing(D:np.ndarray, k:int=5, metric='distance',
                                verbose:int=0, random_state=None,
                                shuffle_equal:bool=True):
    """ Hubness calculations without multiprocessing overhead. """
    log = ConsoleLogging()
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
        if shuffle_equal:
            # Randomize equal values in the distance matrix rows to avoid the
            # problem case if all numbers to sort are the same, which would
            # yield high hubness, even if there is none.
            rp = rnd.permutation(m)
            d2 = d[rp]
            d2idx = np.argpartition(d2, kth=kth)
            D_k[i, :] = rp[d2idx[kth]][::sort_order]
        else:
            d_idx = np.argpartition(d, kth=kth)
            D_k[i, :] = d_idx[kth][::sort_order]

    # N-occurence
    N_k = np.bincount(D_k.astype(int).ravel(), minlength=m)
    # Hubness
    S_k = stats.skew(N_k)

    # return k-hubness, k-nearest neighbors, k-occurence
    if verbose:
        log.message("Hubness calculation done.", flush=True)
    return S_k, D_k, N_k

def hubness_from_vectors(X:np.ndarray, Y:np.ndarray=None, k:int=5,
                         metric='euclidean', verbose:int=0,
                         n_jobs:int=1):
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
        if verbose > 1 or (verbose > 0 and i % 1_000 == 0):
            print(f'Hubness progress {i+1}/{n_test}', end='\r', flush=True)
        d = pairwise_distances(X[i, :].reshape(1, -1), Y, metric, n_jobs)
        nn = np.argpartition(d, kth=kth)[0, start:end]
        Dk[i, :] = nn
    # N-occurence
    Nk = np.bincount(Dk.astype(int).ravel(), minlength=n_train)
    # Hubness
    Sk = stats.skew(Nk)
    return Sk, Dk, Nk


class Hubness(object):
    """ Hubness characteristics of data set.

    Parameters
    ----------
    k : int
        Neighborhood size
    hub_size : float
        Hubs are defined as objects with k-occurrence > hub_size * k.
    metric : string, one of ['euclidean', 'cosine', 'precomputed']
        Metric to use for distance computation. Currently, only
        Euclidean, cosine, and precomputed distances are supported.
    return_k_neighbors : bool
        Whether to save the k-neighbor lists. Requires O(n_test * k) memory.
    return_k_occurrence : bool
        Whether to save the k-occurrence. Requires O(n_test) memory.
    random_state : int, RandomState instance or None, optional
        CURRENTLY IGNORED.
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle_equal : bool, optional
        If true, shuffle neighbors with identical distances to avoid
        artifact hubness.
        NOTE: This is especially useful for secondary distance measures
        with a restricted number of possible values, e.g. SNN or MP empiric.
    n_jobs : int, optional
        CURRENTLY IGNORED.
        Number of processes for parallel computations.
        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs
    verbose : int, optional
        Level of output messages

    Attributes
    ----------
    k_skewness_ : float
        Hubness, measured as skewness of k-occurrence histogram
    k_skewness_truncnom : float
        Hubness, measured as skewness of truncated normal distribution
        fitted with k-occurrence histogram
    antihubs_ : int
        Indices to antihubs
    antihub_occurrence_ : float
        Proportion of antihubs in data set
    hubs_ : int
        Indices to hubs
    hub_occurrence_ : float
        Proportion of k-nearest neighbor slots occupied by hubs
    groupie_ratio_ : float
        Proportion of objects with the largest hub in their neighborhood
    k_occurrence_ : ndarray
        Reverse neighbor count for each object
    k_neighbors_ : ndarray
        Indices to k-nearest neighbors for each object

    References
    ----------
    .. [1] `Author, A. & Author, B. Paper title.  Journal 1:2323 (2020).`
    """

    def __init__(self, k:int=10, hub_size:float=2., metric='euclidean',
                 return_k_neighbors:bool=False,
                 return_k_occurrence:bool=False,
                 verbose:int=0, n_jobs:int=1, random_state=None,
                 shuffle_equal:bool=True, **kwargs):
        self.k = k
        self.hub_size = hub_size
        self.metric = metric
        self.return_k_neighbors = return_k_neighbors
        self.return_k_occurrence = return_k_occurrence
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        self.shuffle_equal = shuffle_equal
        self.kwargs = kwargs

        # Making sure parameters have sensible values
        if k is not None:
            if k < 1:
                raise ValueError(f"Neighborhood size 'k' must "
                                 f"be >= 1, but is {k}.")
        if hub_size <= 0:
            raise ValueError(f"Hub size must be greater than zero.")
        if metric not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. "
                             f"Must be one of {VALID_METRICS}.")
        if not isinstance(return_k_neighbors, bool):
            raise ValueError(f"return_k_neighbors must be True or False.")
        if not isinstance(return_k_occurrence, bool):
            raise ValueError(f"return_k_occurrence must be True or False.")
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"Number of parallel processes 'n_jobs' must be "
                             f"a positive integer, or ``-1`` to use all local"
                             f" CPU cores. Was {n_jobs} instead.")
        if verbose < 0:
            raise ValueError(f"Verbosity level 'verbose' must be >= 0, "
                             f"but was {verbose}.")

    def _k_neighbors(self, X, Y, kth, n_test, start, end):
        assert end - start == self.k, f'Implementation error'
        X_norm = row_norms(X, squared=True)
        Y_norm = row_norms(Y, squared=True)
        try:
            batch_size = self.kwargs['batch_size']
        except KeyError:
            batch_size = 64
        n_batches = np.ceil(n_test / batch_size).astype(int)
        if self.n_jobs == 1:
            Dk = np.zeros((n_test, self.k), dtype=np.int32)
            for i in range(n_batches):
                if self.verbose > 1 \
                    or self.verbose > 0 and (i % 10 == 0 or i+1 == n_batches):
                    log.message(f'k neighbors (from vectors): batch '
                                f'{i+1}/{n_batches} with batch size '
                                f'{batch_size}.', flush=True)
                d = pairwise_distances(
                    X[i*batch_size:(i+1)*batch_size, :], Y, self.metric,
                    self.n_jobs, squared=True, X_norm_squared=X_norm[
                        i*batch_size:(i+1)*batch_size].reshape(1, -1),
                    Y_norm_squared=Y_norm)
                nn = np.argpartition(d, kth=kth)[:, start:end]
                Dk[i*batch_size:(i+1)*batch_size, :] = nn
        else:
            counter = SynchronizedCounter()
            Dk_ctype = RawArray(ctypes.c_int32, n_test * self.k)
            Dk = np.frombuffer(
                Dk_ctype, dtype=np.int32).reshape((n_test, self.k))
            with Pool(processes=self.n_jobs,
                      initializer=_k_neighbors_initializer,
                      initargs=(X, Y, Dk, X_norm, Y_norm, counter)) as pool:
                for _ in pool.map(
                    func=partial(_k_neighbors_parallel,
                                 kth=kth,
                                 start=start,
                                 end=end,
                                 metric=self.metric,
                                 verbose=self.verbose,
                                 batch_size=batch_size,
                                 n_batches=n_batches),
                    chunksize=32,
                    iterable=range(n_batches)):
                    pass # results handled within func
        return Dk

    def _k_neighbors_precomputed(self, D, kth, start, end):
        n_test, m_test = D.shape
        Dk = np.zeros((n_test, self.k), dtype=np.int32)
        for i in range(n_test):
            if self.verbose > 1 \
                or self.verbose and (i % 1000 == 0 or i+1 == n_test):
                log.message(f"k neighbors (from distances): "
                            f"{i+1}/{n_test}.", flush=True)
            d = D[i, :].copy()
            d[~np.isfinite(d)] = np.inf
            if self.shuffle_equal:
                # Randomize equal values in the distance matrix rows to avoid
                # the problem case if all numbers to sort are the same,
                # which would yield high hubness, even if there is none.
                rp = np.random.permutation(m_test)
                d2 = d[rp]
                d2idx = np.argpartition(d2, kth=kth)
                Dk[i, :] = rp[d2idx[start:end]]
            else:
                d_idx = np.argpartition(d, kth=kth)
                Dk[i, :] = d_idx[start:end]
        return Dk

    def _k_neighbors_precomputed_sparse(self, X, n_samples=None):
        ''' Find nearest neighbors in sparse distance matrix. 

        Parameters
        ----------
        X : sparse, shape = [n_test, n_indexed]
            Sparse distance matrix. Only non-zero elements
            may be considered neighbors.
        n_samples : int
            Number of sampled indexed objects, e.g.
            in approximate hubness reduction.
            If None, this is inferred from the first row of X.
    
        Returns
        -------
        k_neighbors : ndarray
            Flattened array of neighbor indices.
        '''
        assert issparse(X), f'Matrix is not sparse'
        X = X.tocsr()
        if n_samples is None:
            n_samples = X.indptr[1] - X.indptr[0]
        n_test, _ = X.shape
        # To allow different number of explicit entries per row,
        # we need to process the matrix row-by-row.
        if np.all(X.indptr[1:] - X.indptr[:-1] == n_samples)\
            and not self.shuffle_equal:
            min_ind = np.argpartition(X.data.reshape(n_test, n_samples),
                                      kth=np.arange(self.k),
                                      axis=1)[:, :self.k]
            k_neighbors = X.indices[
                min_ind.ravel() + np.repeat(X.indptr[:-1], repeats=self.k)]
        else:
            min_ind = np.empty((n_test,), dtype=object)
            k_neighbors = np.empty((n_test,), dtype=object)
            if self.shuffle_equal:
                for i in range(n_test):
                    if self.verbose > 1 \
                        or self.verbose and (i % 1000 == 0 or i+1 == n_test):
                        log.message(f"k neighbors (from sparse distances): "
                                    f"{i+1}/{n_test}.", flush=True)
                    x = X.getrow(i)
                    rp = self.random_state.permutation(x.nnz)
                    d2 = x.data[rp]
                    d2idx = np.argpartition(d2, kth=np.arange(self.k))
                    k_neighbors[i] = x.indices[rp[d2idx[:self.k]]]
            else:
                for i in range(n_test):
                    if self.verbose > 1 \
                        or self.verbose and (i % 1000 == 0 or i+1 == n_test):
                        log.message(f"k neighbors (from sparse distances): "
                                    f"{i+1}/{n_test}.", flush=True)
                    x = X.getrow(i)
                    min_ind = np.argpartition(
                        x.data, kth=np.arange(self.k))[:self.k]
                    k_neighbors[i] = x.indices[min_ind]
            k_neighbors = np.concatenate(k_neighbors)
        return k_neighbors

    def _skewness_truncnorm(self, Nk):
        ''' Corrected hubness measure.
        
        Hubness as skewness of truncated normal distribution
        estimated from k-occurrence histogram.'''
        clip_left = 0
        clip_right = np.iinfo(np.int64).max
        Nk_mean = Nk.mean()
        Nk_std = Nk.std(ddof=1)
        a = (clip_left - Nk_mean) / Nk_std
        b = (clip_right - Nk_mean) / Nk_std
        skew_truncnorm = stats.truncnorm(a, b).moment(3)
        return skew_truncnorm

    def _antihub_occurrence(self, k_occurrence):
        '''Proportion of antihubs in data set.
        
        Antihubs are objects that are never among the nearest neighbors
        of other objects.'''
        antihubs = np.argwhere(k_occurrence == 0).ravel()
        antihub_occurrence = antihubs.size / k_occurrence.size
        return antihubs, antihub_occurrence

    def _hub_occurrence(self, k, k_occurrence, n_test, hub_size=2):
        '''Proportion of nearest neighbor slots occupied by hubs.'''
        hubs = np.argwhere(k_occurrence >= hub_size * k).ravel()
        hub_occurrence = k_occurrence[hubs].sum() / k / n_test
        return hubs, hub_occurrence

    def fit_transform(self, X, Y=None, has_self_distances=False):
        # Let's assume there are no self distances in X
        kth = np.arange(self.k)
        start = 0
        end = self.k
        if self.metric == 'precomputed':
            if Y is not None:
                raise ValueError(
                    f"Y must be None when using precomputed distances.")
            n_test, n_train = X.shape
            if n_test == n_train and has_self_distances:
                kth = np.arange(self.k + 1)
                start = 1
                end = self.k + 1
        else:
            n_test, m_test = X.shape
            if Y is None:
                Y = X
                # Self distances do occur in this case
                kth = np.arange(self.k + 1)
                start = 1
                end = self.k + 1
            n_train, m_train = Y.shape
            assert m_test == m_train, f'Number of features do not match'

        if self.metric == 'precomputed':
            if issparse(X):
                k_neighbors = self._k_neighbors_precomputed_sparse(X)
            else:
                k_neighbors = self._k_neighbors_precomputed(X, kth, start, end)
        else:
            k_neighbors = self._k_neighbors(
                X, Y, kth=kth, n_test=n_test, start=start, end=end)
        if self.return_k_neighbors:
            self.k_neighbors_ = k_neighbors
        k_occurrence = np.bincount(
            k_neighbors.astype(int).ravel(), minlength=n_train)
        if self.return_k_occurrence:
            self.k_occurrence_ = k_occurrence
        # traditional skewness measure
        self.k_skewness_ = stats.skew(k_occurrence)
        # corrected skewness measure (against truncated normal distribution)
        self.k_skewness_truncnorm_ = self._skewness_truncnorm(k_occurrence)
        # anti-hub occurrence
        self.antihubs_, self.antihub_occurrence_ = \
            self._antihub_occurrence(k_occurrence)
        # hub occurrence
        self.hubs_, self.hub_occurrence_ = \
            self._hub_occurrence(k=self.k, k_occurrence=k_occurrence,
                                 n_test=n_test, hub_size=self.hub_size)
        # Largest hub
        # TODO That should probably also be diveded by k...
        self.groupie_ratio_ = k_occurrence.max() / n_test
        return self


if __name__ == '__main__':
    # Simple test case
    from hub_toolbox.io import load_dexter
    dexter_distance, l, v = load_dexter()
    Sn, Dk, Nk = hubness(dexter_distance)
    Snv, Dkv, Nkv = hubness_from_vectors(v, metric='cosine')
    print("Hubness =", Sn, Snv)
