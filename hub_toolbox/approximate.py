#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
Licensed under the terms of the GNU GPLv3.

(c) 2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import ctypes
from functools import partial
import warnings
from multiprocessing import cpu_count, RawArray, Pool

import numpy as np
from scipy.sparse.base import issparse
from scipy.stats import norm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import euclidean_distances
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import cosine_distances

try:
    import nmslib
    nms_avail = True
except ImportError:
    nms_avail = False
try:
    import falconn
    falconn_avail = True
except ImportError:
    falconn_avail = False

VALID_HR = ["MP", "MPG", "LS", "NICDM", "DSL"]
VALID_SAMPLE = ['random', 'kmeans++', 'lsh', 'hnsw', None]
VALID_METRICS = ['sqeuclidean', 'cosine']

def enforce_finite_distances(arr):
    ''' Replace non-finite distances with the max. distance'''
    fin_ind = np.isfinite(arr)
    nonfin_ind = ~fin_ind
    # Replace, iff there are any non-finite distances
    if nonfin_ind.any():
        if fin_ind.any():
            d_max = arr[fin_ind].max()
        else:
            d_max = 1
        arr[nonfin_ind] = d_max
    return arr

#==============================================================================
# HELPER FUNCTIONS FOR MULTIPROCESSING
#==============================================================================
def _load_shared_data(X_train_=None, X_test_=None,
                      ind_train_=None, ind_test_=None,
                      D_train_=None, D_test_=None,
                      D_sec_=None, ann_index_=None):
    global X_train, X_test, ind_train, ind_test, D_train, D_test, D_sec
    global ann_index
    X_train = X_train_
    X_test = X_test_
    ind_train = ind_train_
    ind_test = ind_test_
    D_train = D_train_
    D_test = D_test_
    D_sec = D_sec_
    ann_index = ann_index_

def _shared_lsh(X_train_, ind_train_, D_train_, lsh_index_, num_probes):
    global X_train, ind_train, D_train, ann_index
    X_train = X_train_
    ind_train = ind_train_
    D_train = D_train_
    ann_index = lsh_index_.construct_query_object()
    ann_index.set_num_probes(num_probes)

def _shared_lsh_trafo(X_train_, X_test_, ind_test_, D_test_,
                      lsh_index_, num_probes):
    global X_train, X_test, ind_test, D_test, ann_index
    X_train = X_train_
    X_test = X_test_
    ind_test = ind_test_
    D_test = D_test_
    ann_index = lsh_index_.construct_query_object()
    ann_index.set_num_probes(num_probes)

def _lsh_filt(i, k, metric, verbose):
    # LSH will find object itself as 1-NN
    x = X_train[i, :]
    knn = np.array(ann_index.find_k_nearest_neighbors(x, k=k+1))[1:]
    ind_train[i, :knn.size] = knn
    if metric == 'sqeucliean':
        D_train[i, :knn.size] = euclidean_distances(
            x.reshape(1, -1), X_train[knn], squared=True)
    elif metric == 'cosine':
        D_train[i, :knn.size] = cosine_distances(
            x.reshape(1, -1), X_train[knn])
    if knn.size < k:
        ind_train[i, knn.size:] = knn[-1]
        D_train[i, knn.size:] = D_train[i].max()

def _lsh_trafo(i, k, metric, verbose):
    x = X_test[i, :]
    lsh_nn = np.array(ann_index.find_k_nearest_neighbors(x, k=k))
    if metric == 'sqeucliean':
        D_test[i, :lsh_nn.size] = euclidean_distances(
            x.reshape((1, -1)), X_train[lsh_nn], squared=True).ravel()
    elif metric == 'cosine':
        D_test[i, :lsh_nn.size] = cosine_distances(
            x.reshape((1, -1)), X_train[lsh_nn])
    ind_test[i, :lsh_nn.size] = lsh_nn
    if lsh_nn.size < k:
        ind_test[i, lsh_nn.size:] = lsh_nn[-1]
    D_test[i, lsh_nn.size:] = D_test[i].max()

def _mp_hnsw(i, n_train, n_test, n_samples, verbose):
    if verbose > 1 and (i % 1000 == 0 or i == n_test-1):
        print(f"MP_empiric: {i+1} of {n_test}.", end='\r', flush=True)
    for j in range(n_samples):
        d = D_test[i, j]
        t_idx = ind_test[i, j]
        # O(1) instead of O(n_samples)
        ind_x = ind_test[i, :j+1]
        # O(log n_samples) instead of O(n_samples)
        thresh_ind = np.searchsorted(D_train[t_idx], d, side='right')
        ind_t = ind_train[t_idx, :thresh_ind]
        mp_complement = np.union1d(ind_x, ind_t)
        mp_ind = np.setdiff1d(range(n_train), mp_complement, assume_unique=True)
        D_sec[i, j] = 1 - mp_ind.size / n_train

def _mp_lsh(i, n_train, n_test, n_samples, verbose):
    if verbose > 1 and (i % 1000 == 0 or i == n_test-1):
        print(f"MP_empiric: {i+1} of {n_test}.", end='\r', flush=True)
    x = X_test[i, :]
    for j in range(n_samples):
        d = D_test[i, j]
        t = X_train[ind_test[i, j], :]
        # So far, only FALCONN is supported
        ind_x = ann_index.find_near_neighbors(query=x, threshold=d)
        ind_t = ann_index.find_near_neighbors(query=t, threshold=d)
        mp_complement = np.union1d(ind_x, ind_t)
        mp_ind = np.setdiff1d(range(n_train), mp_complement, assume_unique=True)
        D_sec[i, j] = 1 - mp_ind.size / n_train

def _mp_full(i, n_train, n_test, n_samples, verbose):
    if verbose > 1 and (i % 1000 == 0 or i == n_test-1):
        print(f"MP_empiric: {i+1} of {n_test}.", end='\r', flush=True)
    dI = D_test[i, :][np.newaxis, :] # broadcasted afterwards
    dJ = D_train#[self.train_ind_, :] # fancy indexing, thus copy
    d = dI.T # D[i, :][:, np.newaxis] # both versions are equal
    # div by n
    n_pts = n_samples
    D_sec[i, :] = 1 - (np.sum((dI > d) & (dJ > d), axis=1) / n_pts)

#==============================================================================
# INSTANCE SELECTION ALGORITHMS
#==============================================================================
def kmeanspp(X, n_clusters, x_squared_norms=None, random_state=None,
             n_local_trials=None, return_ind=True, metric='sqeuclidean'):
    """Select vantage points according to k-means++

    Parameters
    -----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to select vantage points from. To avoid memory copy, the input
        data should be double precision (dtype=np.float64).
    n_clusters : integer
        The number of vantage points to choose
    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : numpy.RandomState
        The generator used to initialize the centers.
    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    return_ind : bool, default = True
        Return the indices of cluster centers w.r.t. X
    Notes
    -----
    Adapted from scikit-learn code (under BSD-3 license).
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f'Invalid metric "{metric}" in kmeanspp(). '
                         f'Must be one of {VALID_METRICS}.')
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    ind = np.empty((n_clusters,), dtype=np.int32)

    if x_squared_norms is None and metric == 'sqeuclidean':
        x_squared_norms = row_norms(X, squared=True)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if issparse(X):
        centers[0] = X[center_id].toarray()
        ind[0] = center_id
    else:
        centers[0] = X[center_id]
        ind[0] = center_id

    # Initialize list of closest distances and calculate current potential
    if metric == 'sqeuclidean':
        closest_dist_sq = euclidean_distances(
            centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
            squared=True)
    elif metric == 'cosine':
        closest_dist_sq = cosine_distances(centers[0, np.newaxis], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)

        # Compute distances to center candidates
        if metric == 'sqeuclidean':
            distance_to_candidates = euclidean_distances(
                X[candidate_ids], X, Y_norm_squared=x_squared_norms, 
                squared=True)
        elif metric == 'cosine':
            distance_to_candidates = cosine_distances(X[candidate_ids], X)

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq,
                                     distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        # Permanently add best center candidate found in local tries
        if issparse(X):
            centers[c] = X[best_candidate].toarray()
            ind[c] = best_candidate
        else:
            centers[c] = X[best_candidate]
            ind[c] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    if return_ind:
        return centers, ind
    else:
        return centers

class SuQHR(BaseEstimator, TransformerMixin):
    """ Approximate hubness reduction with subquadratic complexity.

    Parameters
    ----------
    hr_algorithm : string, one of ['MP', 'MPG', 'LS', 'NICDM', 'DSL']
        Hubness reduction algorithm. 
        
        * MP: Mutual proximity with empiric distance distribution
        * MPG: Mutual proximity with Gaussian distance distr.
        * LS: Local scaling (with parameter k = 'n_neighbors')
        * NICDM: Non-iterative contextual dissimilarity measure
                 (with parameter k = 'n_neighbors')
        * DSL: DisSim Local (with parameter kappa = 'n_neighbors')
    n_neighbors : int
        Neighborhood size (used e.g. in local scaling, and DisSimLocal).
    n_samples : int
        Distances are calculated to `n_samples` objects.
        Depending on 'sampling_algorithm', these objects are either
        identical for all test objects, or dependent on test objects.
        NOTE: It is crucial to keep this fixed (independent of number
        of objects), in order to achieve true subquadratic complexity.
    sampling_algorithm : string, one of ['random', 'kmeans++', 'lsh', 'hnsw']
        Algorithm to use for nearest neighbors search.
        Choose 'lsh' or 'hnsw' for approximate NN search: In these cases,
        'n_samples' near neighbors are selected in a filtering step, and
        further refined with hubness reduction.
        Otherwise, fixed vantage points are selected 'random'ly, or according
        to more sophisticated instance selection schemes (currently only
        'kmeans++').
        If None, compute full distance matrices (FuQHR)
    metric : string, one of ['sqeuclidean', 'cosine']
        Metric to use for distance computation. Currently, only squared
        Euclidean distances are supported (GCD of all used libraries).
    random_state : int, RandomState instance or None, optional
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``sampling_algorithm`` == 'random'.
    n_jobs : int, optional
        Number of processes for parallel computations.
        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs
    verbose : int, optional
        Level of output messages
    kwargs : any
        Additional named arguments

    Attributes
    ----------
    ...
    fixed_vantage_pts_ : bool
        True: Random, kmeans++ sampling use the same training objects
        as vantage points for all test objects.
        False: LSH, HNSW filtering yield different vantage points
        for each test object
    X_train_norm_squared_ : ndarray
        Precomputed squared norm for training objects to speed up
        Euclidean distance calculation at test time.
    ann_index_ : object (LSH.query or HNSW.index)
        Approx. nearest neighbor index structure for fast query at test time.
    X_train_ : ndarray
        Training vectors
    y_train_ : ndarray of integers
        Training labels corresponding to TODO describe
    ind_train_ : ndarray of integers
        TODO describe
    mu_train_, sd_train_ : ndarray (for hr_algorithm == 'MPG')
        Mean and standard deviation of training distances
    References
    ----------
    .. [1] `Author, A. & Author, B. Paper title.  Journal 1:2323 (2020).`
    """

    def __init__(self, hr_algorithm='mp', n_neighbors=5, n_samples=100,
                 sampling_algorithm='random', metric='sqeuclidean',
                 random_state=None, n_jobs=1, verbose=0, **kwargs):
        self.hr_algorithm = hr_algorithm
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.sampling_algorithm = sampling_algorithm
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.kwargs = kwargs

        # Making sure parameters have sensible values
        if hr_algorithm is not None and hr_algorithm.upper() not in VALID_HR:
            raise ValueError(
                f'Unknown hubness reduction algorithm "{hr_algorithm}". '
                f'Must be one of {VALID_HR}.')
        if n_neighbors is not None:
            if n_neighbors < 1:
                raise ValueError(f"Neighborhood size 'n_neighbors' must "
                                 f"be >= 1, but is {n_neighbors}.")
        if n_samples is not None:
            if n_samples < 1:
                raise ValueError(f"Sample size 'n_samples' must "
                                 f"be >= 1, but is {n_samples}. "
                                 f"Try a value in [100..1000].")
        if sampling_algorithm is not None:
            if sampling_algorithm not in VALID_SAMPLE:
                raise ValueError(
                    f'Unknown sampling algorithm "{sampling_algorithm}". '
                    f'Must be one of {VALID_SAMPLE}.')
            elif sampling_algorithm == 'lsh' and not falconn_avail:
                raise ImportError(f'LSH not available. Please make sure '
                                  f'FALCONN is installed.')
            elif sampling_algorithm == 'hnsw' and not nms_avail:
                raise ImportError(f'HNSW not available. Please make sure '
                                  f'NMSLIB is installed.')
        if metric not in VALID_METRICS:
            raise ValueError(f"Unknown metric '{metric}'. "
                             f"Must be one of {VALID_METRICS}.")
        if n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs < -1 or n_jobs == 0:
            raise ValueError(f"Number of parallel processes 'n_jobs' must be "
                             f"a positive integer, or ``-1`` to use all local"
                             f" CPU cores. Was {n_jobs} instead.")
        if verbose < 0:
            raise ValueError(f"Verbosity level 'verbose' must be >= 0, "
                             f"but was {verbose}.")


    def _X_norm_squared_if_sqeuclidean(self, X):
        if self.metric == 'sqeuclidean':
            X_norm_squared = row_norms(X, squared=True)
        else:
            X_norm_squared = None
        return X_norm_squared

    ############################################################################
    ##
    ##  Sampling and filtering methods 
    ##
    def _random_sampling(self, X, y=None):
        random_state = check_random_state(self.random_state)
        n_classes = np.unique(y).size # Also works for y=None
        if y is None or n_classes > self.n_samples:
            if n_classes > self.n_samples:
                warnings.warn(
                    f'For stratified random sampling n_samples = '
                    f'{self.n_samples} must be greater or equal to the '
                    f'number of classes = {n_classes}. Resorting to '
                    f'non-stratified sampling. Some classes will be lost!')
            shuffle = ShuffleSplit(
                n_splits=1, test_size=self.n_samples, random_state=random_state)
        else:
            shuffle = StratifiedShuffleSplit(
                n_splits=1, test_size=self.n_samples, random_state=random_state)
        _, ind = next(shuffle.split(X=X, y=y))
        X = X[ind, :]
        if y is not None:
            y = y[ind]
        X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
        return X, y, ind, X_norm_squared

    def _kmeanspp_sampling(self, X, y=None):
        random_state = check_random_state(self.random_state)
        # precompute squared norms of data points
        if self.metric == 'sqeuclidean':
            x_squared_norms = row_norms(X, squared=True)
        else:
            x_squared_norms = None
        X, ind = kmeanspp(X, n_clusters=self.n_samples, 
                          x_squared_norms=x_squared_norms,
                          random_state=random_state,
                          return_ind=True)
        y = y[ind]
        if self.metric == 'sqeuclidean':
            x_squared_norms = x_squared_norms[ind]
        return X, y, ind, x_squared_norms

    def _hnsw_filtering(self, X, k):
        ''' Find k approx. nearest neighbors for each vector in X with HNSW.
        
        Returns indices of neighbors per object, their distances,
        and the HNSW index.'''
        # Initialize a new index, using a HNSW index on Eucl distances
        # or cosine similarities.
        n_train = X.shape[0]
        try:
            method = self.kwargs['nms__method']
            post = self.kwargs['nms__post']
        except KeyError:
            method = 'hnsw'
            post = 2
        if self.metric == 'sqeuclidean':
            space = 'l2'
        elif self.metric == 'cosine':
            space = 'cosinesimil'
        self.hnsw_squared_euclidean_ = True if space == 'l2' else False
        self.hnsw_cosine_simil_ = True if space == 'cosinesimil' else False
        assert not (self.hnsw_cosine_simil_
                    and self.hnsw_squared_euclidean_),\
            f'Implementation error in _hnsw_filtering'
        index = nmslib.init(method=method, space=space)
        index.addDataPointBatch(X)
        index.createIndex({'post': post}, print_progress=(self.verbose > 2))
        # ANN search will find object itself as 1-NN
        neigh_dist = index.knnQueryBatch(
            X, k=k+1, num_threads=self.n_jobs)
        ind = np.zeros((n_train, k), dtype=np.int32) * n_train
        D_train = np.empty_like(ind) * np.nan
        for i, (idx, dist) in enumerate(neigh_dist):
            ind[i, :idx.size-1] = idx[1:]
            D_train[i, :dist.size-1] = dist[1:]
            # If not enough neighbors were found:
            if idx.size - 1 < k:
                ind[i, idx.size-1:] = idx[-1]
                D_train[i, dist.size-1:] = dist.max()
        if self.hnsw_squared_euclidean_:
            D_train **= 2
        elif self.hnsw_cosine_simil_:
            # convert to cosine distances
            D_train *= -1
            D_train += 1
        return ind, D_train, index

    def _hnsw_trafo(self, X, k):
        n_test = X.shape[0]
        if self.verbose > 2:
            print(f'HR.transform(hnsw)')
        neigh_dist = self.ann_index_.knnQueryBatch(
            X, k=k, num_threads=self.n_jobs)
        ind = np.zeros((n_test, k), dtype=np.int32) * self.X_train_.shape[0]
        D_test = np.empty_like(ind) * np.nan
        for i, (idx, dist) in enumerate(neigh_dist):
            ind[i, :idx.size] = idx
            D_test[i, :dist.size] = dist
            if idx.size < k:
                ind[i, idx.size:] = idx[-1]
                D_test[i, dist.size:] = dist.max()
        if self.hnsw_squared_euclidean_:
            D_test **= 2
        elif self.hnsw_cosine_simil_:
            D_test *= -1
            D_test += 1
        return ind, D_test

    def _lsh_filtering(self, X, k):
        ''' Find k approximate nearest neighbors for each vector in X with LSH.
        
        Returns indices of neighbors per object, their distances,
        and the LSH index.'''
        n_train = X.shape[0]
        if self.metric == 'sqeuclidean':
            distance = falconn.DistanceFunction.EuclideanSquared  # @UndefinedVariable
        elif self.metric == 'cosine':
            distance = falconn.DistanceFunction.NegativeInnerProduct  # @UndefinedVariable
        try:
            num_probes = self.kwargs['falconn__num_probes']
        except KeyError:
            num_probes = 50
        self.num_probes_ = num_probes
        # Set up the LSH index
        lsh_cp = falconn.get_default_parameters(*X.shape, distance=distance)
        lsh_index = falconn.LSHIndex(lsh_cp)
        lsh_index.setup(X)
        # Constructing a query object
        query = lsh_index.construct_query_object()
        query.set_num_probes(num_probes)
        if self.n_jobs == 1:
            ind = np.empty((n_train, k), dtype=np.int32)
            D_train = np.empty_like(ind, dtype=np.float32)
            for i, x in enumerate(X):
                # LSH will find object itself as 1-NN
                knn = np.array(query.find_k_nearest_neighbors(x, k=k+1))[1:]
                ind[i, :knn.size] = knn
                if self.metric == 'sqeuclidean':
                    D_train[i, :knn.size] = euclidean_distances(
                        x.reshape(1, -1), X[knn], squared=True)
                elif self.metric == 'cosine':
                    D_train[i, :knn.size] = cosine_distances(
                        x.reshape(1, -1), X[knn])
                if knn.size < k:
                    ind[i, knn.size:] = knn[-1]
                    D_train[i, knn.size:] = D_train[i].max()
        else:
            ind_ctype = RawArray(ctypes.c_int32, n_train * k)
            ind = np.frombuffer(ind_ctype, dtype=np.int32).reshape((n_train, k))
            D_train_ctype = RawArray(ctypes.c_float, ind.size)
            D_train = np.frombuffer(D_train_ctype, dtype=np.float32).reshape(ind.shape)
            with Pool(processes=self.n_jobs,
                      initializer=_shared_lsh,
                      initargs=(X, ind, D_train, lsh_index, num_probes)) as pool:
                    for _ in pool.map(
                        func=partial(_lsh_filt, k=k, metric=self.metric,
                                     verbose=self.verbose),
                        iterable=range(n_train)):
                        pass # results handled within func
        self.lsh_index_ = lsh_index
        return ind, D_train, query

    def _lsh_trafo(self, X, k):
        n_test = X.shape[0]
        if self.verbose > 2:
            print(f'HR.transform(lsh)')
        metric = self.metric
        if self.n_jobs == 1:
            D_test = np.empty((n_test, k), dtype=X.dtype)
            ind = np.empty_like(D_test, dtype=np.int32)
            for i, x in enumerate(X):
                lsh_nn = np.array(
                    self.ann_index_.find_k_nearest_neighbors(x, k=k))
                if metric == 'sqeuclidean':
                    D_test[i, :lsh_nn.size] = euclidean_distances(
                        x.reshape((1, -1)), self.X_train_[lsh_nn],
                        squared=True).ravel()
                elif metric == 'cosine':
                    D_test[i, :lsh_nn.size] = cosine_distances(
                        x.reshape((1, -1)), self.X_train_[lsh_nn]).ravel()
                ind[i, :lsh_nn.size] = lsh_nn
                if lsh_nn.size < k:
                    ind[i, lsh_nn.size:] = lsh_nn[-1]
                    D_test[i, lsh_nn.size:] = D_test[i].max()
        else:
            ind_ctype = RawArray(ctypes.c_int32, n_test * k)
            ind = np.frombuffer(
                ind_ctype, dtype=np.int32).reshape((n_test, k))
            D_test_ctype = RawArray(ctypes.c_float, ind.size)
            D_test = np.frombuffer(
                D_test_ctype, dtype=np.float32).reshape(ind.shape)
            with Pool(processes=self.n_jobs,
                      initializer=_shared_lsh_trafo,
                      initargs=(self.X_train_, X, ind, D_test, self.lsh_index_,
                                self.num_probes_)) as pool:
                    for _ in pool.map(
                        func=partial(_lsh_trafo, k=k, metric=metric,
                                     verbose=self.verbose),
                        iterable=range(n_test)):
                        pass # results handled within func
        return ind, D_test

    ############################################################################
    ##
    ##  Baseline without hubness reduction
    ##
    def _fit_without_hr(self, X, y=None):
        X = check_array(X, dtype=float)
        k = self.n_samples
        # Sampling
        if self.sampling_algorithm in ['random', 'kmeans++', None]:
            if self.sampling_algorithm == 'random':
                X, y, ind, X_norm_squared = self._random_sampling(X, y)
            elif self.sampling_algorithm == 'kmeans++':
                X, y, ind, X_norm_squared = self._kmeanspp_sampling(X, y)
            elif self.sampling_algorithm is None:
                n_train = X.shape[0]
                ind = np.tile(
                    np.arange(n_train), n_train).reshape((n_train, n_train))
                X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
            self.fixed_vantage_pts_ = True
        # Approximate Nearest Neighbor Filtering
        elif self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO: use in HNSW, LSH filtering
            X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
            if self.sampling_algorithm == 'hnsw':
                ind, _, hnsw_index = self._hnsw_filtering(X, k)
                self.ann_index_ = hnsw_index
            if self.sampling_algorithm == 'lsh':
                ind, _, lsh_index = self._lsh_filtering(X, k)
                self.ann_index_ = lsh_index
            self.fixed_vantage_pts_ = False
        else:
            raise NotImplementedError(
                f'Value for `sampling_algorithm` must be one of '
                f'{VALID_SAMPLE}. NOTE: This error indicates a software bug.')
        self.X_train_ = X
        self.y_train_ = y
        self.ind_train_ = ind
        self.X_train_norm_squared_ = X_norm_squared
        return

    def _transform_without_hr(self, X):
        n_test, _ = X.shape
        n_train = self.X_train_.shape[0]
        k = self.n_samples
        # TODO use self.X_train_norm_squared_
        if self.sampling_algorithm == 'hnsw':
            ind, D_test = self._hnsw_trafo(X, k)
            self.ind_test_ = ind
        elif self.sampling_algorithm == 'lsh':
            ind, D_test = self._lsh_trafo(X, k)
            self.ind_test_ = ind
        else:
            if self.verbose > 2:
                print(f'LS.transform(full)')
            if self.metric == 'sqeuclidean':
                D_test = euclidean_distances(
                    X=X, Y=self.X_train_,
                    Y_norm_squared=self.X_train_norm_squared_, squared=True)
            elif self.metric == 'cosine':
                D_test = cosine_distances(X=X, Y=self.X_train_)
            ind = np.tile(np.arange(n_train), n_test).reshape((n_test, n_train))
            self.ind_test_ = self.ind_train_
        self.sec_dist_ = D_test
        return None

    ############################################################################
    ##
    ##  Mutual proximity using empiric distance distributions ('exact' MP)
    ##
    def _fit_mp(self, X, y=None):
        X = check_array(X, dtype=float)
        k = self.n_samples
        n_train, _ = X.shape
        # Sampling
        if self.sampling_algorithm in ['random', 'kmeans++', None]:
            if self.sampling_algorithm == 'random':
                X, y, ind, X_norm_squared = self._random_sampling(X, y)
            elif self.sampling_algorithm == 'kmeans++':
                X, y, ind, X_norm_squared = self._kmeanspp_sampling(X, y, )
            elif self.sampling_algorithm is None:
                ind = np.arange(n_train) # use all objects
                X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
                self.n_samples = n_train
            self.X_train_norm_squared_ = X_norm_squared
            self.fixed_vantage_pts_ = True
        # Approximate Nearest Neighbor Filtering
        elif self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO add X_nrom_squared
            if self.sampling_algorithm == 'hnsw':
                ind, D_train, ann_index = self._hnsw_filtering(X, k)
                self.D_train_ = D_train
            if self.sampling_algorithm == 'lsh':
                ind, D_train, ann_index = self._lsh_filtering(X, k)
            self.ann_index_ = ann_index
            self.fixed_vantage_pts_ = False
        else:
            raise NotImplementedError(
                f'Value for `sampling_algorithm` must be one of '
                f'{VALID_SAMPLE}. NOTE: This error indicates a software bug.')
        self.X_train_ = X
        self.y_train_ = y
        self.ind_train_ = ind
        return

    def _transform_mp(self, X):
        n_test, _ = X.shape
        k = self.n_samples
        n_train = self.X_train_.shape[0]
        if self.sampling_algorithm == 'hnsw':
            ind, D_test = self._hnsw_trafo(X, k)
            self.ind_test_ = ind
            # Calculate MP empiric
            D_sec_ctype = RawArray(ctypes.c_double, D_test.size)
            D_sec = np.frombuffer(D_sec_ctype, dtype=np.float64).reshape(D_test.shape)
            with Pool(processes=self.n_jobs,
                      initializer=_load_shared_data,
                      initargs=(None, None, self.ind_train_, ind,
                                self.D_train_, D_test, D_sec)) as pool:
                for _ in pool.map(
                    func=partial(_mp_hnsw, n_train=n_train, n_test=n_test,
                                 n_samples=k, verbose=self.verbose),
                    iterable=range(n_test)):
                    pass # results handled within func
            self.sec_dist_ = D_sec
        elif self.sampling_algorithm == 'lsh':
            ind, D_test = self._lsh_trafo(X, k)
            self.ind_test_ = ind
            # Calculate MP empiric
            if self.n_jobs == 1:
                D_sec = np.empty_like(D_test)
                for i in range(n_test):
                    if self.verbose > 1 and (i % 1000 == 0 or i == n_test-1):
                        print(f"MP_empiric: {i+1} of {n_test}.", end='\r', flush=True)
                    x = X[i, :]
                    for j in range(self.n_samples):
                        d = D_test[i, j]
                        t = self.X_train_[ind[i, j], :]
                        # So far, only FALCONN is supported
                        ind_x = self.ann_index_.find_near_neighbors(query=x, threshold=d)
                        ind_t = self.ann_index_.find_near_neighbors(query=t, threshold=d)
                        mp_complement = np.union1d(ind_x, ind_t)
                        mp_ind = np.setdiff1d(range(n_train), mp_complement, assume_unique=True)
                        D_sec[i, j] = 1 - mp_ind.size / n_train
            else:
                D_sec_ctype = RawArray(ctypes.c_double, D_test.size)
                D_sec = np.frombuffer(D_sec_ctype, dtype=np.float64).reshape(D_test.shape)
                with Pool(processes=self.n_jobs,
                      initializer=_load_shared_data,
                      initargs=(self.X_train_, X, None, ind,
                                None, D_test, D_sec, self.ann_index_)) as pool:
                    for _ in pool.map(
                        func=partial(_mp_lsh, n_train=n_train, n_test=n_test,
                                     n_samples=k, verbose=self.verbose),
                        iterable=range(n_test)):
                        pass # results handled within func
            self.sec_dist_ = D_sec
        else:
            if self.verbose > 2:
                print(f'HR.transform(full)')
            if self.metric == 'sqeuclidean':
                D_test = euclidean_distances(X, self.X_train_,
                    Y_norm_squared=self.X_train_norm_squared_, squared=True)
                D_train = euclidean_distances(self.X_train_,
                    Y_norm_squared=self.X_train_norm_squared_, squared=True)
            elif self.metric == 'cosine':
                D_test = cosine_distances(X, self.X_train_)
                D_train = cosine_distances(self.X_train_)
            np.fill_diagonal(D_train, np.inf)
            # Calculate MP empiric
            if self.n_jobs == 1:
                D_sec = np.empty_like(D_test)
                for i in range(n_test):
                    if self.verbose > 1 and (i % 1000 == 0 or i == n_test-1):
                        print(f"MP_empiric: {i+1} of {n_test}.",
                              end='\r', flush=True)
                    dI = D_test[i, :][np.newaxis, :] # broadcasted afterwards
                    dJ = D_train
                    d = dI.T
                    # div by n
                    n_pts = self.n_samples
                    D_sec[i, :] = \
                        1 - (np.sum((dI > d) & (dJ > d), axis=1) / n_pts)
            else:
                D_sec_ctype = RawArray(ctypes.c_double, D_test.size)
                D_sec = np.frombuffer(
                    D_sec_ctype, dtype=np.float64).reshape(D_test.shape)
                with Pool(processes=self.n_jobs,
                      initializer=_load_shared_data,
                      initargs=(None, None, None, None,
                                D_train, D_test, D_sec)) as pool:
                    for _ in pool.map(
                        func=partial(_mp_full, n_train=n_train, n_test=n_test,
                                     n_samples=k, verbose=self.verbose),
                        iterable=range(n_test)):
                        pass # results handled within func
            self.sec_dist_ = D_sec
        return

    ############################################################################
    ##
    ##  Mutual proximity assuming independent Gaussian distance distributions
    ##
    def _fit_mpg(self, X, y=None):
        X = check_array(X, dtype=float)
        n_train, _ = X.shape
        k = self.n_samples
        # Sampling
        if self.sampling_algorithm in ['random', 'kmeans++', None]:
            if self.sampling_algorithm == 'random':
                X, y, ind, X_norm_squared = self._random_sampling(X, y)
            elif self.sampling_algorithm == 'kmeans++':
                X, y, ind, X_norm_squared = self._kmeanspp_sampling(X, y)
            elif self.sampling_algorithm is None:
                ind = np.arange(n_train) # use all objects
                X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
                self.n_samples = n_train
            if self.metric == 'sqeuclidean':
                D_train = euclidean_distances(
                    X, X_norm_squared=X_norm_squared.reshape(1, -1),
                    squared=True)
            elif self.metric == 'cosine':
                D_train = cosine_distances(X)
            np.fill_diagonal(D_train, np.nan)
            self.mu_train_ = np.nanmean(D_train, axis=0)
            self.sd_train_ = np.nanstd(D_train, axis=0, ddof=0)
            np.fill_diagonal(D_train, 0)
            self.X_train_norm_squared_ = X_norm_squared
            self.fixed_vantage_pts_ = True
        # Approximate Nearest Neighbor Filtering
        elif self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO use X_norm_squared_
            if self.sampling_algorithm == 'hnsw':
                ind, D_train, hnsw_index = self._hnsw_filtering(X, k)
                self.ann_index_ = hnsw_index
            if self.sampling_algorithm == 'lsh':
                ind, D_train, lsh_index = self._lsh_filtering(X, k)
                self.ann_index_ = lsh_index
            self.fixed_vantage_pts_ = False
        else:
            raise NotImplementedError(
                f'Value for `sampling_algorithm` must be one of '
                f'{VALID_SAMPLE}. NOTE: This error indicates a software bug.')
        self.X_train_ = X
        self.y_train_ = y
        self.ind_train_ = ind
        
        if self.sampling_algorithm is not None:
            self.mu_train_ = np.mean(D_train, axis=1)
            self.sd_train_ = np.std(D_train, axis=1, ddof=0)
        return

    def _transform_mpg(self, X):
        n_test, _ = X.shape
        n_train = self.X_train_.shape[0]
        k = self.n_samples
        if self.sampling_algorithm == 'hnsw':
            ind, D_test = self._hnsw_trafo(X, k)
            self.ind_test_ = ind
        elif self.sampling_algorithm == 'lsh':
            ind, D_test = self._lsh_trafo(X, k)
            self.ind_test_ = ind
        else:
            if self.verbose > 2:
                print(f'LS.transform(full)')
            if self.metric == 'sqeuclidean':
                D_test = euclidean_distances(
                    X=X, Y=self.X_train_, 
                    Y_norm_squared=self.X_train_norm_squared_, squared=True)
            elif self.metric == 'cosine':
                D_test = cosine_distances(X=X, Y=self.X_train_)
            ind = np.tile(np.arange(n_train), n_test).reshape((n_test, n_train))
            self.ind_test_ = self.ind_train_
        # Calculate MP G
        D_mp = np.empty_like(D_test)
        mu_train = self.mu_train_
        sd_train = self.sd_train_
        for i in range(n_test):
            if self.verbose > 1 and (i % 1000 == 0 or i == n_test-1):
                    print(f"MP_Gaussian: {i+1} of {n_test}.", end='\r', flush=True)
            j_mom = ind[i]
            mu = np.nanmean(D_test[i])
            sd = np.nanstd(D_test[i], ddof=0)
            p1 = norm.sf(D_test[i, :], mu, sd)
            p2 = norm.sf(D_test[i, :], mu_train[j_mom], sd_train[j_mom])
            D_mp[i, :] = (1 - p1 * p2).ravel()
        self.sec_dist_ = D_mp
        return None


    ############################################################################
    ##
    ##  Local scaling / NICDM
    ##
    def _fit_ls(self, X, y=None):
        X = check_array(X, dtype=float)
        k = self.n_samples
        kth = self.n_neighbors
        # Sampling
        if self.sampling_algorithm in ['random', 'kmeans++', None]:
            if self.sampling_algorithm == 'random':
                X, y, ind, X_norm_squared = self._random_sampling(X, y)
            elif self.sampling_algorithm == 'kmeans++':
                X, y, ind, X_norm_squared = self._kmeanspp_sampling(X, y)
            elif self.sampling_algorithm is None:
                ind = None
                X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
            if self.metric == 'sqeuclidean':
                D_train = euclidean_distances(
                    X, Y_norm_squared=X_norm_squared, squared=True)
            elif self.metric == 'cosine':
                D_train = cosine_distances(X)
            self.r_train_ = np.partition(D_train, kth=kth)[:, 1:kth+1]
            self.X_train_norm_squared_ = X_norm_squared
            self.fixed_vantage_pts_ = True
        # Approximate Nearest Neighbor Filtering
        elif self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO use X_norm_squared
            if self.sampling_algorithm == 'hnsw':
                ind, D_train, ann_index = self._hnsw_filtering(X, k)
            if self.sampling_algorithm == 'lsh':
                ind, D_train, ann_index = self._lsh_filtering(X, k)
            self.ann_index_ = ann_index
            self.r_train_ = D_train[:, :kth] # self distances filtered by ann_filtering() methods
            self.fixed_vantage_pts_ = False
        else:
            raise NotImplementedError(
                f'Value for `sampling_algorithm` must be one of '
                f'{VALID_SAMPLE}. NOTE: This error indicates a software bug.')
        self.X_train_ = X
        self.y_train_ = y
        self.ind_train_ = ind
        return

    def _transform_ls(self, X):
        n_test, _ = X.shape
        n_train = self.X_train_.shape[0]
        k = self.n_samples
        kth = self.n_neighbors - 1
        if self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO use X_nrom_squared
            if self.sampling_algorithm == 'hnsw':
                ind, D_test = self._hnsw_trafo(X, k)
            elif self.sampling_algorithm == 'lsh':
                ind, D_test = self._lsh_trafo(X, k)
            self.ind_test_ = ind
            r_test = D_test[:, :self.n_neighbors] # already sorted
        else:
            if self.verbose > 2:
                print(f'HR.transform(full)')
            if self.metric == 'sqeuclidean':
                D_test = euclidean_distances(
                    X, self.X_train_,
                    Y_norm_squared=self.X_train_norm_squared_, squared=True)
            elif self.metric == 'cosine':
                D_test = cosine_distances(X=X, Y=self.X_train_)
            ind = np.tile(
                np.arange(n_train), n_test).reshape((n_test, n_train))
            self.ind_test_ = self.ind_train_
            r_test = np.partition(D_test, kth=kth)[:, :self.n_neighbors]
        # Calculate LS or NICDM
        D_sec = np.empty_like(D_test)
        if not self.fixed_vantage_pts_:
            sample_ind = self.ind_test_
            assert sample_ind.shape[0] == n_test, \
                (f'sample_ind.shape={sample_ind.shape} '
                 f'incompatible with D_test.shape={D_test.shape}')
        if self.hr_algorithm.upper() == 'LS':
            r_train = self.r_train_[:, kth]
            r_test = r_test[:, kth]
            for i in range(n_test):
                if self.verbose > 1 and (i % 1000 == 0 or i == n_test-1):
                    print(f"Local scaling: {i+1} of {n_test}.",
                          end='\r', flush=True)
                if self.fixed_vantage_pts_:
                    D_sec[i, :] = 1 - np.exp(-1 * D_test[i]**2 \
                                             / (r_test[i] * r_train[:]))
                else:
                    D_sec[i, :] = \
                        1 - np.exp(-1 * D_test[i]**2 \
                                   / (r_test[i] * r_train[sample_ind[i]]))
        elif self.hr_algorithm.upper() == 'NICDM':
            r_train = self.r_train_.mean(axis=1)
            r_test = r_test.mean(axis=1)
            for i in range(n_test):
                if self.verbose > 1 and (i % 1000 == 0 or i == n_test-1):
                    print(f"NICDM: {i+1} of {n_test}.", end='\r', flush=True)
                if self.fixed_vantage_pts_:
                    D_sec[i, :] = D_test[i] / np.sqrt((r_test[i] * r_train[:]))
                else:
                    D_sec[i, :] = D_test[i] / np.sqrt((r_test[i] * r_train[sample_ind[i]]))
        else:
            raise ValueError(f"Invalid 'hr_algorithm' {self.hr_algorithm} "
                             f"in '_transform_ls()'.")
        self.sec_dist_ = D_sec
        return None


    ############################################################################
    ##
    ##  DisSim Local
    ##
    def _fit_dsl(self, X, y=None):
        X = check_array(X, dtype=float)
        kth = self.n_neighbors
        # Sampling
        if self.sampling_algorithm in ['random', 'kmeans++', None]:
            if self.sampling_algorithm == 'random':
                X, y, ind, X_norm_squared = self._random_sampling(X, y)
            elif self.sampling_algorithm == 'kmeans++':
                X, y, ind, X_norm_squared = self._kmeanspp_sampling(X, y)
            elif self.sampling_algorithm is None:
                ind = None
                X_norm_squared = self._X_norm_squared_if_sqeuclidean(X)
            # Local centroids of training objects
            if self.metric == 'sqeuclidean':
                D_train = euclidean_distances(
                    X, Y_norm_squared=X_norm_squared, squared=True)
            elif self.metric == 'cosine':
                # A rather exploratory approach...
                D_train = cosine_distances(X)
            knn = np.argpartition(D_train, kth=kth+1)[:, 1:kth+1]
            del D_train
            centroids = np.empty_like(X)
            for i in range(X.shape[0]):
                centroids[i, :] = X[knn[i, :], :].mean(axis=0)
            self.X_train_norm_squared_ = X_norm_squared
            self.fixed_vantage_pts_ = True
        # Approximate Nearest Neighbor Filtering
        elif self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO use X_norm_squared
            if self.sampling_algorithm == 'hnsw':
                ind, D_train, ann_index = \
                    self._hnsw_filtering(X, self.n_samples)
                #D_train **= 2 # Squared Eucl. distances for DSL
            if self.sampling_algorithm == 'lsh':
                ind, D_train, ann_index = \
                    self._lsh_filtering(X, self.n_samples)
            self.ann_index_ = ann_index
            self.fixed_vantage_pts_ = False
            # Local centroids of training objects
            centroids = np.empty_like(X)
            for i in range(X.shape[0]): # ind argsorts D_train
                centroids[i, :] = X[ind[i, :kth], :].mean(axis=0)
        else:
            raise NotImplementedError(
                f'Value for `sampling_algorithm` must be one of '
                f'{VALID_SAMPLE}. NOTE: This error indicates a software bug.')
        if self.metric == 'sqeuclidean':
            dist_to_cent = row_norms(X - centroids, squared=True) #((X - centroids) ** 2).sum(axis=1)
        elif self.metric == 'cosine':
            dist_to_cent = 1 - (X * centroids).sum(axis=1) \
                            / np.linalg.norm(X, ord=2, axis=1) \
                            / np.linalg.norm(centroids, ord=2, axis=1)
        self.X_train_centroids_ = centroids
        self.X_train_dist_to_cent_ = dist_to_cent
        self.X_train_ = X
        self.y_train_ = y
        self.ind_train_ = ind
        return

    def _transform_dsl(self, X):
        n_test, _ = X.shape
        n_train = self.X_train_.shape[0]
        k = self.n_samples
        kth = self.n_neighbors
        if self.sampling_algorithm in ['hnsw', 'lsh']:
            # TODO use X_norm_squared
            if self.sampling_algorithm == 'hnsw':
                ind, D_test = self._hnsw_trafo(X, k)
            elif self.sampling_algorithm == 'lsh':
                ind, D_test = self._lsh_trafo(X, k)
            self.ind_test_ = ind
            centroid_test = np.empty_like(X)
            for i in range(n_test):
                centroid_test[i, :] = \
                    self.X_train_[ind[i, :kth], :].mean(axis=0)
        else:
            if self.verbose > 2:
                print(f'HR.transform(full)')
            if self.metric == 'sqeuclidean':
                D_test = euclidean_distances(
                    X, self.X_train_,
                    Y_norm_squared=self.X_train_norm_squared_, squared=True)
            elif self.metric == 'cosine':
                D_test = cosine_distances(X=X, Y=self.X_train_)
            ind = np.tile(np.arange(n_train), n_test).reshape((n_test, n_train))
            self.ind_test_ = self.ind_train_
            centroid_test = np.empty_like(X)
            knn = np.argpartition(D_test, kth=kth)[:, :kth]
            for i in range(n_test):
                centroid_test[i, :] = self.X_train_[knn[i, :], :].mean(axis=0)
        # DisSim Local
        if self.metric == 'sqeuclidean':
            X_test = X - centroid_test
            X_test **= 2
            X_test_dist_to_cent = X_test.sum(axis=1)
        elif self.metric == 'cosine':
            X_test_dist_to_cent = \
                1. - (X * centroid_test).sum(axis=1) \
                    / np.linalg.norm(X, ord=2, axis=1) \
                    / np.linalg.norm(centroid_test, ord=2, axis=1)
        X_train_dist_to_cent = np.empty_like(ind, dtype=float)
        for i in range(n_test):
            X_train_dist_to_cent[i] = self.X_train_dist_to_cent_[ind[i]] # TODO check correct use of ind
        D_sec = D_test
        D_sec -= X_test_dist_to_cent[:, np.newaxis]
        D_sec -= X_train_dist_to_cent
        self.sec_dist_ = D_sec
        return None


    ############################################################################
    ##
    ##  General methods - fit, predict, etc.
    ##
    def fit(self, X, y=None):
        """Setup the nearest neighbor index and/or sampling and/or filtering.

        Parameters
        ----------
        X : array-like of shape [n_objects, n_features]
            training set.
        y: array-like of shape [n_objects,]
            Class labels (used for stratified random sampling)

        Returns
        -------
        self : returns an instance of self.
        """
        if self.hr_algorithm is None:
            fit = self._fit_without_hr
        else:
            hr = self.hr_algorithm.upper()
            if hr == 'MP':
                fit = self._fit_mp
            elif hr == 'MPG':
                fit = self._fit_mpg
            elif hr in ['LS', 'NICDM']:
                fit = self._fit_ls
            elif hr == 'DSL':
                fit = self._fit_dsl
            else:
                raise ValueError(
                    f'Unknown hubness reduction algorithm "{hr}". '
                    f'Must be one of ["MP", "MPG", "LS", "NICDM", "DSL"].')
        fit(X, y)
        return self

    def transform(self, X):
        """ Transform new points into embedding space.

        Parameters
        ----------
        X : array-like, shape = [n_objects, n_features]
            Test set.

        Returns
        -------
        D_ls : array, shape [n_objects, n_samples]

        Notes
        -----
        ...
        """
        X = check_array(X)
        # TODO use all attributes, that are required for 
        # the individual transformation.
        if self.hr_algorithm is None:
            check_is_fitted(self, ["X_train_"])
            transform = self._transform_without_hr
        else:
            hr = self.hr_algorithm.upper()
            if hr == 'MP':
                check_is_fitted(self, ["X_train_"])
                transform = self._transform_mp
            elif hr == 'MPG':
                check_is_fitted(self, ["X_train_"])
                transform = self._transform_mpg
            elif hr in ['LS', 'NICDM']:
                check_is_fitted(self, ["X_train_"])
                transform = self._transform_ls
            elif hr == 'DSL':
                check_is_fitted(self, ["X_train_"])
                transform = self._transform_dsl
            else:
                raise NotImplementedError(
                    f'Transforming with hr_algorithm "{hr}" '
                    f'is not (yet) implemented.')
        transform(X)
        return self.sec_dist_

class ApproximateHubnessReduction(SuQHR):
    pass
