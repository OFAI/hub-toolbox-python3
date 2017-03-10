#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2016, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import numpy as np
from scipy.sparse.base import issparse
from scipy.sparse.lil import lil_matrix
from hub_toolbox import IO, Logging

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
    log = Logging.ConsoleLogging()
    # Checking input
    IO._check_sample_shape_fits(D, train_ind)
    IO._check_valid_metric_parameter(metric)
    sparse = issparse(D)
    if metric == 'similarity':
        if train_ind is not None:
            raise NotImplementedError
        sort_order = -1
        exclude = -np.inf
        self_value = 1.
        log.warning("Similarity matrix support for LS is experimental.")
    else: # metric == 'distance':
        sort_order = 1
        exclude = np.inf
        self_value = 0
        if sparse:
            log.error("Sparse distance matrices are not supported.")
            raise NotImplementedError(
                "Sparse distance matrices are not supported.")

    D = np.copy(D)
    n = D.shape[0]
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
        nn = np.argsort(di)[::sort_order]
        r[i] = di[nn[k-1]] #largest similarities or smallest distances

    if sparse:
        D_ls = lil_matrix(D.shape)
        # Number of nonzero cells per row
        nnz = np.array([x.size for x in np.split(D.indices, D.indptr[1:-1])])
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

def local_scaling(D:np.ndarray, k:int=7, metric:str='distance',
                  test_ind:np.ndarray=None):
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
    log = Logging.ConsoleLogging()
    # Checking input
    IO._check_distance_matrix_shape(D)
    IO._check_valid_metric_parameter(metric)
    sparse = issparse(D)
    if metric == 'similarity':
        sort_order = -1
        exclude = -np.inf
        self_tmp_value = np.inf
        self_value = 1.
        log.warning("Similarity matrix support for LS is experimental.")
    else: # metric == 'distance':
        sort_order = 1
        exclude = np.inf
        self_value = 0
        self_tmp_value = self_value
        if sparse:
            log.error("Sparse distance matrices are not supported.")
            raise NotImplementedError(
                "Sparse distance matrices are not supported.")

    D = np.copy(D)
    n = D.shape[0]
    if test_ind is None:
        train_set_ind = slice(0, n) #take all        
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_ind)

    r = np.zeros(n)
    for i in range(n):
        if sparse:
            di = D[i, train_set_ind].toarray()
        else:
            di = D[i, train_set_ind]
        di[i] = exclude
        nn = np.argsort(di)[::sort_order]
        r[i] = di[nn[k-1]] #largest similarities or smallest distances

    if sparse:
        D_ls = lil_matrix(D.shape)
        # Number of nonzero cells per row
        nnz = np.array([x.size for x in np.split(D.indices, D.indptr[1:-1])])
    else:
        D_ls = np.zeros_like(D)

    for i in range(n):
        # vectorized inner loop: calc only triu part
        tmp = np.empty(n-i)
        tmp[0] = self_tmp_value
        if metric == 'similarity':
            if sparse and nnz <= k:     # Don't rescale if there are
                tmp[1:] = D[i, i+1:]    # too few neighbors in row
            else:
                tmp[1:] = np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
        else:
            tmp[1:] = 1 - np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
        D_ls[i, i:] = tmp
    # copy triu to tril -> symmetric matrix (diag=zeros)
    # NOTE: does not affect self values, since inf+inf=inf and 0+0=0
    D_ls += D_ls.T

    if sparse:
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
    IO._check_sample_shape_fits(D, train_ind)
    IO._check_valid_metric_parameter(metric)
    if metric == 'similarity':
        raise NotImplementedError("NICDM does not support similarity matrices "
                                  "at the moment.")
    else: # metric == 'distance':
        D = np.copy(D)
        sort_order = 1
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
    r = np.zeros(n)
    for i in range(n):
        di = D[i, :]
        nn = np.argsort(di)[::sort_order]
        knn[i, :] = di[nn[0:k]] # largest sim. or smallest dist.
        r[i] = np.mean(knn[i]) 
    r_geom = _local_geomean(knn.ravel())

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

def nicdm(D:np.ndarray, k:int=7, metric:str='distance',
          test_ind:np.ndarray=None):
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
    #log = Logging.ConsoleLogging()
    # Checking input
    IO._check_distance_matrix_shape(D)
    IO._check_valid_metric_parameter(metric)
    if metric == 'similarity':
        raise NotImplementedError("NICDM does not support similarity matrices "
                                  "at the moment.")
    D = np.copy(D)

    if metric == 'distance':
        sort_order = 1
        exclude = np.inf
    else: #metric == 'similarity':
        sort_order = -1
        exclude = -np.inf

    n = D.shape[0]

    if test_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_ind)

    knn = np.zeros((n, k))
    r = np.zeros(n)
    np.fill_diagonal(D, np.inf)
    for i in range(n):
        di = D[i, :].copy()
        di[i] = exclude
        di = di[train_set_ind]
        nn = np.argsort(di)[::sort_order]
        knn[i, :] = di[nn[0:k]] # largest sim. or smallest dist.
        r[i] = np.mean(knn[i])
    r_geom = _local_geomean(knn.ravel())

    D_nicdm = np.zeros_like(D)
    for i in range(n):
        # vectorized inner loop for 100x speed-up (using broadcasting)
        #D_nicdm[i, i+1:] = ((r_geom**2) * D[i, i+1:]) / (r[i] * r[i+1:])
        D_nicdm[i, i+1:] = (r_geom * D[i, i+1:]) / np.sqrt(r[i] * r[i+1:])
    D_nicdm += D_nicdm.T

    return D_nicdm

def _local_geomean(x):
    return np.exp(np.sum(np.log(x)) / np.max(np.shape(x)))
