#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2017, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import ctypes
from functools import partial
from multiprocessing import cpu_count, RawArray, Pool
import numpy as np
from scipy.sparse.base import issparse
from scipy.sparse.lil import lil_matrix
from hub_toolbox import io, logging

__all__ = ['local_scaling', 'local_scaling_sample', 'nicdm', 'nicdm_sample']

def local_scaling_sample(D:np.ndarray, k:int=7, metric:str='distance',
                         train_ind:np.ndarray=None, test_ind:np.ndarray=None):
    """Transform a distance matrix with Local Scaling.

    --- DRAFT version ---

    Transforms the given distance matrix into new one using local scaling [1]_
    with the given `k`-th nearest neighbor. There are two types of local
    scaling methods implemented. The original one and NICDM, both reduce
    hubness in distance spaces, similarly to Mutual Proximity.

    Parameters
    ----------
    D : ndarray or csr_matrix
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 7)
        Neighborhood radius for local scaling.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

        NOTE: self similarities in sparse `D_ls` are set to ``np.inf``

    train_ind : ndarray, optional
        If given, use only these data points as neighbors for rescaling.

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    Returns
    -------
    D_ls : ndarray
        Secondary distance LocalScaling matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    log = logging.ConsoleLogging()
    # Checking input
    io.check_sample_shape_fits(D, train_ind)
    io.check_valid_metric_parameter(metric)
    sparse = issparse(D)
    n = D.shape[0]
    if metric == 'similarity':
        if train_ind is not None:
            raise NotImplementedError
        kth = n - k
        exclude = -np.inf
        self_value = 1.
        log.warning("Similarity matrix support for LS is experimental.")
    else: # metric == 'distance':
        kth = k - 1
        exclude = np.inf
        self_value = 0
        if sparse:
            log.error("Sparse distance matrices are not supported.")
            raise NotImplementedError(
                "Sparse distance matrices are not supported.")

    D = np.copy(D)
    if test_ind is None:
        train_set_ind = slice(0, n) #take all
        n_ind = range(n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_ind)
        n_ind = test_ind
    # Exclude self distances
    for j, sample in enumerate(train_ind):
        D[sample, j] = exclude
    r = np.zeros(n)
    for i in range(n):
        if train_ind is None:
            if sparse:
                di = D[i, train_set_ind].toarray()
            else:
                di = D[i, train_set_ind]
        else:
            di = D[i, :] # all columns are training in this case
        r[i] = np.partition(di, kth=kth)[kth]

    if sparse:
        D_ls = lil_matrix(D.shape)
        # Number of nonzero cells per row
        nnz = D.getnnz(axis=1)
    else:
        D_ls = np.zeros_like(D)

    if metric == 'similarity':
        for i in n_ind:
            if sparse and nnz[i] <= k: # Don't rescale if there are too few 
                D_ls[i, :] = D[i, :]   # neighbors in the current row
            else:
                D_ls[i, :] = np.exp(-1 * D[i, :]**2 / (r[i] * r[train_ind]))
    else:
        for i in n_ind:
            D_ls[i, :] = 1 - np.exp(-1 * D[i, :]**2 / (r[i] * r[train_ind]))

    if test_ind is None:
        if sparse:
            return D_ls.tocsr()
        else:
            np.fill_diagonal(D_ls, self_value)
            return D_ls
    else:
        # Ensure correct self distances
        for j, sample in enumerate(train_ind):
            D_ls[sample, j] = self_value
        return D_ls[test_ind]

#===============================================================================
# #=============================================================================
# #                             LOCAL SCALING
# #=============================================================================
#===============================================================================

def _ls_load_shared_data(D_, train_ind_, r_, r_ctype_,
                         D_ls_=None, D_ls_ctype_=None):
    global D, train_ind, r, r_ctype, D_ls, D_ls_ctype
    D = D_
    train_ind = train_ind_
    r = r_
    r_ctype = r_ctype_
    D_ls = D_ls_
    D_ls_ctype = D_ls_ctype_
    return

def _ls_calculate_r(i, kth):
    di = D[i, train_ind]
    r[i] = np.partition(di, kth=kth)[kth]
    return

def _ls_calculate_sec_dist(i, n, metric, self_tmp_value):
    # vectorized inner loop: calc only triu part
    tmp = np.empty(n-i)
    tmp[0] = self_tmp_value
    if metric == 'similarity':
        tmp[1:] = np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
    else:
        tmp[1:] = 1 - np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
    D_ls[i, i:] = tmp
    D_ls[i:, i] = tmp
    return

def local_scaling(D:np.ndarray, k:int=7, metric:str='distance',
                  test_ind:np.ndarray=None, n_jobs:int=1):
    """Transform a distance matrix with Local Scaling.

    Transforms the given distance matrix into new one using local scaling [1]_
    with the given `k`-th nearest neighbor. There are two types of local
    scaling methods implemented. The original one and NICDM, both reduce
    hubness in distance spaces, similarly to Mutual Proximity.

    Parameters
    ----------
    D : ndarray or csr_matrix
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 7)
        Neighborhood radius for local scaling.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

        NOTE: self similarities in sparse `D_ls` are set to ``np.inf``

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    n_jobs : int, optional, default: 1
        Number of processes for parallel computations.

        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs

    Returns
    -------
    D_ls : ndarray
        Secondary distance LocalScaling matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    log = logging.ConsoleLogging()
    # Checking input
    io.check_distance_matrix_shape(D)
    io.check_valid_metric_parameter(metric)
    sparse = issparse(D)
    n = D.shape[0]
    if n_jobs == -1:
        n_jobs = cpu_count()
    if metric == 'similarity':
        kth = n - k
        exclude = -np.inf
        self_tmp_value = np.inf
        self_value = 1.
        log.warning("Similarity matrix support for LS is experimental.")
        if sparse and n_jobs != 1:
            log.warning("Parallel processing not implemented for sparse "
                        "matrices. Using single process instead.")
            n_jobs = 1
    else: # metric == 'distance':
        kth = k - 1
        exclude = np.inf
        self_value = 0
        self_tmp_value = self_value
        if sparse:
            log.error("Sparse distance matrices are not supported.")
            raise NotImplementedError(
                "Sparse distance matrices are not supported.")
    D = np.copy(D)

    if test_ind is None:
        train_ind = slice(0, n) #take all        
    else:
        train_ind = np.setdiff1d(np.arange(n), test_ind)
    if sparse:
        r = np.zeros(n)
        for i in range(n):
            di = D[i, train_ind].toarray()
            di[i] = exclude
            r[i] = np.partition(di, kth=kth)[kth]
        D_ls = lil_matrix(D.shape)
        # Number of nonzero cells per row
        nnz = D.getnnz(axis=1)
    else:
        np.fill_diagonal(D, exclude)
        if n_jobs > 1:
            r_ctype = RawArray(ctypes.c_double, n)
            r = np.frombuffer(r_ctype, dtype=np.float64)
            with Pool(processes=n_jobs,
                      initializer=_ls_load_shared_data,
                      initargs=(D, train_ind, r, r_ctype)) as pool:
                for _ in pool.imap(func=partial(_ls_calculate_r, kth=kth),
                                   iterable=range(n)):
                    pass # results handled within func
        else:
            r = np.partition(D[:, train_ind], kth=kth)[:, kth]

    if sparse or n_jobs == 1:
        D_ls = np.zeros_like(D)
        for i in range(n):
            # vectorized inner loop: calc only triu part
            tmp = np.empty(n-i)
            tmp[0] = self_tmp_value
            if metric == 'similarity':
                if sparse and nnz[i] <= k:  # Don't rescale if there are
                    tmp[1:] = np.nan        # too few neighbors in row
                else:
                    tmp[1:] = np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
            else:
                tmp[1:] = 1 - np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
            D_ls[i, i:] = tmp
        # copy triu to tril -> symmetric matrix (diag=zeros)
        # NOTE: does not affect self values, since inf+inf=inf and 0+0=0
        D_ls += D_ls.T
    else:
        D_ls_ctype = RawArray(ctypes.c_double, D.size)
        D_ls = np.frombuffer(D_ls_ctype, dtype=np.float64).reshape(D.shape)
        with Pool(processes=n_jobs,
                  initializer=_ls_load_shared_data,
                  initargs=(D, train_ind, r, r_ctype, D_ls, D_ls_ctype)) as pool:
            for _ in pool.imap(func=partial(_ls_calculate_sec_dist,
                                  n=n, metric=metric,
                                  self_tmp_value=self_tmp_value),
                               iterable=range(n)):
                pass # results handled within func
        # triu is copied to tril within func
    if sparse:
        for i, nz in enumerate(nnz):
            if nz <= k: # too few neighbors
                D_ls[i, :] = D[i, :]
        return D_ls.tocsr()
    else:
        np.fill_diagonal(D_ls, self_value)
        return D_ls

def nicdm_sample(D:np.ndarray, k:int=7, metric:str='distance',
                 train_ind:np.ndarray=None, test_ind:np.ndarray=None):
    """Transform a distance matrix with local scaling variant NICDM.
    
    --- DRAFT version ---

    Transforms the given distance matrix into new one using NICDM [1]_
    with the given neighborhood radius `k` (average). There are two types of
    local scaling methods implemented. The original one and the non-iterative
    contextual dissimilarity measure, both reduce hubness in distance spaces,
    similarly to Mutual Proximity.

    Parameters
    ----------
    D : ndarray or csr_matrix
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 7)
        Neighborhood radius for local scaling.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix `D` is a distance or similarity matrix.

        NOTE: self similarities in sparse `D_ls` are set to ``np.inf``

    train_ind : ndarray, optional
        If given, use only these data points as neighbors for rescaling.

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    Returns
    -------
    D_nicdm : ndarray
        Secondary distance NICDM matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    # Checking input
    io.check_sample_shape_fits(D, train_ind)
    io.check_valid_metric_parameter(metric)
    if metric == 'similarity':
        raise NotImplementedError("NICDM does not support similarity matrices "
                                  "at the moment.")
    else: # metric == 'distance':
        D = np.copy(D)
        kth = np.arange(k)
        exclude = np.inf
        self_value = 0
        if issparse(D):
            raise NotImplementedError(
                "Sparse distance matrices are not supported.")

    n = D.shape[0]
    if test_ind is None:
        n_ind = range(n)
    else:
        n_ind = test_ind
    # Exclude self distances
    for j, sample in enumerate(train_ind):
        D[sample, j] = exclude

    # Statistics
    knn = np.zeros((n, k))
    r = np.partition(D, kth=kth, axis=1)[:, :k].mean(axis=1)
    r_geom = _local_geomean(r) #knn.ravel())

    # Calculate secondary distances
    D_nicdm = np.zeros_like(D)
    for i in n_ind:
        # vectorized inner loop (using broadcasting)
        D_nicdm[i, :] = (r_geom * D[i, :]) / np.sqrt(r[i] * r[train_ind])
        #D_nicdm[i, :] = ((r_geom**2) * D[i, :]) / (r[i] * r[train_ind])

    # Ensure correct self distances and return sec. dist. matrix
    if test_ind is None:
        np.fill_diagonal(D_nicdm, self_value)
        return D_nicdm 
    else:
        for j, sample in enumerate(train_ind):
            D_nicdm[sample, j] = self_value
        return D_nicdm[test_ind]


#==============================================================================
# #============================================================================
# #             Non-iterative Contextual Dissimilarity Measure
# #============================================================================
#==============================================================================
def _nicdm_load_shared_data(D_, train_ind_, r_, r_ctype_,
                            D_nicdm_=None, D_nicdm_ctype_=None):
    global D, train_ind, r, r_ctype, D_nicdm, D_nicdm_ctype
    D = D_
    train_ind = train_ind_
    r = r_
    r_ctype = r_ctype_
    D_nicdm = D_nicdm_
    D_nicdm_ctype = D_nicdm_ctype_
    return

def _nicdm_calculate_r(i, kth, k):
    di = D[i, train_ind]
    r[i] = np.partition(di, kth=kth)[:k].mean()
    return

def _nicdm_calculate_sec_dist(i, r_geom, n, metric):
    # vectorized inner loop
    tmp = (r_geom * D[i, i+1:]) / np.sqrt(r[i] * r[i+1:])
    D_nicdm[i, i+1:] = tmp
    D_nicdm[i+1:, i] = tmp
    return

def nicdm(D:np.ndarray, k:int=7, metric:str='distance',
          test_ind:np.ndarray=None, n_jobs:int=1):
    """Transform a distance matrix with local scaling variant NICDM.

    Transforms the given distance matrix into new one using NICDM [1]_
    with the given neighborhood radius `k` (average). There are two types of
    local scaling methods implemented. The original one and the non-iterative
    contextual dissimilarity measure, both reduce hubness in distance spaces,
    similarly to Mutual Proximity.

    Parameters
    ----------
    D : ndarray
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 7)
        Neighborhood radius for local scaling.

    metric : {'distance'}, optional (default: 'distance')
        Currently, only distance matrices are supported.

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    n_jobs : int, optional, default: 1
        Number of processes for parallel computations.

        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs

    Returns
    -------
    D_nicdm : ndarray
        Secondary distance NICDM matrix.

    References
    ----------
    .. [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012).
           Local and global scaling reduce hubs in space. The Journal of Machine
           Learning Research, 13(1), 2871–2902.
    """
    #log = logging.ConsoleLogging()
    # Checking input
    io.check_distance_matrix_shape(D)
    io.check_valid_metric_parameter(metric)
    if metric == 'similarity':
        raise NotImplementedError("NICDM does not support similarity matrices "
                                  "at the moment.")
    else:
        kth = np.arange(k)
        exclude = np.inf
    if n_jobs == -1:
        n_jobs = cpu_count()
    D = np.copy(D)
    n = D.shape[0]

    if test_ind is None:
        train_ind = slice(0, n)
    else:
        train_ind = np.setdiff1d(np.arange(n), test_ind)

    np.fill_diagonal(D, exclude)
    if n_jobs > 1:
        r_ctype = RawArray(ctypes.c_double, n)
        r = np.frombuffer(r_ctype, dtype=np.float64)
        with Pool(processes=n_jobs,
                  initializer=_nicdm_load_shared_data,
                  initargs=(D, train_ind, r, r_ctype)) as pool:
            for i, knn in enumerate(pool.imap(
                func=partial(_nicdm_calculate_r, kth=kth, k=k),
                iterable=range(n))):
                pass # r is handled within func
            r_geom = _local_geomean(r)
        D_nicdm_ctype = RawArray(ctypes.c_double, D.size)
        D_nicdm = np.frombuffer(D_nicdm_ctype, dtype=np.float64).reshape(D.shape)
        with Pool(processes=n_jobs,
                  initializer=_nicdm_load_shared_data,
                  initargs=(D, train_ind, r, r_ctype, D_nicdm, D_nicdm_ctype)) as pool:
            for _ in pool.imap(
                func=partial(_nicdm_calculate_sec_dist, r_geom=r_geom, n=n, metric=metric),
                iterable=range(n)):
                pass # results handled within func
    else: # no multiprocessing
        knn = np.partition(D[:, train_ind], kth=kth, axis=1)[:, :k]
        r = knn.mean(axis=1)
        r_geom = _local_geomean(r)

        D_nicdm = np.zeros_like(D)
        for i in range(n):
            # vectorized inner loop for 100x speed-up (using broadcasting)
            #D_nicdm[i, i+1:] = ((r_geom**2) * D[i, i+1:]) / (r[i] * r[i+1:])
            D_nicdm[i, i+1:] = (r_geom * D[i, i+1:]) / np.sqrt(r[i] * r[i+1:])
        D_nicdm += D_nicdm.T

    return D_nicdm

def _local_geomean(x):
    return np.exp(np.sum(np.log(x)) / np.max(np.shape(x)))
