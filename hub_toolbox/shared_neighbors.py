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
import ctypes
from functools import partial
from multiprocessing import cpu_count, Pool, RawArray
import numpy as np
from hub_toolbox import io

__all__ = ['snn_sample', 'shared_nearest_neighbors', 'simhub', 'simhubIN']

#==============================================================================
# #============================================================================
# #                             SNN SAMPLE
# #============================================================================
#==============================================================================

def _snns_init(D_, knn_, train_ind_, D_snn_):
    global distance, knn, train_ind, D_snn
    distance = D_
    knn = knn_
    train_ind = train_ind_
    D_snn = D_snn_
    return

def _snns_my_hood(i, k, sort_order):
    di = distance[i, :]
    # TODO change to np.partition for PERF
    nn = np.argsort(di)[::sort_order]
    knn[i, nn[0:k]] = True
    return

def _snns_our_hood(i, k, metric):
    knn_i = knn[i, :]
    # using broadcasting
    Dij = np.sum(np.logical_and(knn_i, knn[train_ind, :]), 1)
    if metric == 'distance':
        D_snn[i, :] = 1. - Dij / k
    else: # metric == 'similarity':
        D_snn[i, :] = Dij / k
    return

def snn_sample(D:np.ndarray, k:int=10, metric='distance',
               train_ind:np.ndarray=None, test_ind:np.ndarray=None,
               n_jobs:int=1):
    """Transform distance matrix using shared nearest neighbors [1]_.

    __DRAFT_VERSION__

    SNN similarity is based on computing the overlap between the `k` nearest
    neighbors of two objects. SNN approaches try to symmetrize nearest neighbor
    relations using only rank and not distance information [2]_.

    Parameters
    ----------
    D : np.ndarray
        The ``n x s`` distance (similarity) matrix, where ``s==train_ind.size``

    k : int, optional (default: 10)
        Neighborhood radius: The `k` nearest neighbors are used to calculate SNN.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether the matrix `D` is a distance or similarity matrix

    train_ind : ndarray, optional
        If given, use only these data points as neighbors for rescaling.

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
    D_snn : ndarray
        Secondary distance SNN matrix

    References
    ---------- 
    .. [1] R. Jarvis and E. A. Patrick, “Clustering using a similarity measure
           based on shared near neighbors,” IEEE Transactions on Computers,
           vol. 22, pp. 1025–1034, 1973.

    .. [2] Flexer, A., & Schnitzer, D. (2013). Can Shared Nearest Neighbors
           Reduce Hubness in High-Dimensional Spaces? 2013 IEEE 13th
           International Conference on Data Mining Workshops, 460–467.
           http://doi.org/10.1109/ICDMW.2013.101
    """
    io.check_sample_shape_fits(D, train_ind)
    io.check_valid_metric_parameter(metric)
    if metric == 'distance':
        self_value = 0.
        sort_order = 1
        exclude = np.inf
    if metric == 'similarity':
        self_value = 1.
        sort_order = -1
        exclude = -np.inf
    distance = D.copy()
    n = distance.shape[0]
    if test_ind is None:
        n_ind = range(n)
    else:
        n_ind = test_ind
    # Exclude self distances
    for j, sample in enumerate(train_ind):
        distance[sample, j] = exclude

    if n_jobs == -1:
        n_jobs = cpu_count()
    if n_jobs > 1:
        knn_ctype = RawArray(ctypes.c_bool, distance.size)
        knn = np.frombuffer(knn_ctype, dtype=bool).reshape(D.shape)
        D_snn_ctype = RawArray(ctypes.c_double, distance.size)
        D_snn = np.frombuffer(D_snn_ctype, dtype=np.float64).reshape(D.shape)
        with Pool(processes=n_jobs,
                  initializer=_snns_init,
                  initargs=(distance, knn, train_ind, D_snn)) as pool:
            for _ in pool.imap(
                func=partial(_snns_my_hood, k=k, sort_order=sort_order),
                iterable=range(n)):
                pass # Handling inside function
            for _ in pool.imap(
                func=partial(_snns_our_hood, k=k, metric=metric),
                iterable=n_ind):
                pass # Handling inside function
    else:
        knn = np.zeros_like(distance, bool)
        # find nearest neighbors for each point
        for i in range(n):
            di = distance[i, :]
            # TODO change to np.partition for PERF
            nn = np.argsort(di)[::sort_order]
            knn[i, nn[0:k]] = True
        D_snn = np.zeros_like(distance)
        for i in n_ind:
            knn_i = knn[i, :]
            # using broadcasting
            Dij = np.sum(np.logical_and(knn_i, knn[train_ind, :]), 1)
            if metric == 'distance':
                D_snn[i, :] = 1. - Dij / k
            else: # metric == 'similarity':
                D_snn[i, :] = Dij / k

    # Ensure correct self distances and return sec. dist. matrix
    if test_ind is None:
        np.fill_diagonal(D_snn, self_value)
        return D_snn
    else:
        for j, sample in enumerate(train_ind):
            D_snn[sample, j] = self_value
        return D_snn[test_ind]

#==============================================================================
# #============================================================================
# #                         SHARED NEAREST NEIGHBORS
# #============================================================================
#==============================================================================

def _snn_init(distance_, knn_, D_snn_):
    global distance, knn, D_snn
    distance = distance_
    knn = knn_
    D_snn = D_snn_
    return

def _snn_my_hood(i, k, kth, sort_order):
    di = distance[i, :]
    nn = np.argpartition(di, kth=kth)[::sort_order]
    knn[i, nn[:k]] = True
    return

def _snn_our_hood(i, k, metric):
    knn_i = knn[i, :]
    # using broadcasting
    Dij = np.sum(np.logical_and(knn_i, knn[i+1:, :]), 1)
    if metric == 'distance':
        D_snn[i, i+1:] = 1. - Dij / k
    else: # metric == 'similarity':
        D_snn[i, i+1:] = Dij / k
    return

def shared_nearest_neighbors(D:np.ndarray, k:int=10, metric='distance',
                             n_jobs:int=1):
    """Transform distance matrix using shared nearest neighbors [1]_.

    SNN similarity is based on computing the overlap between the `k` nearest
    neighbors of two objects. SNN approaches try to symmetrize nearest neighbor
    relations using only rank and not distance information [2]_.

    Parameters
    ----------
    D : np.ndarray
        The ``n x n`` symmetric distance (similarity) matrix.

    k : int, optional (default: 10)
        Neighborhood radius: The `k` nearest neighbors are used to calculate SNN.

    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether the matrix `D` is a distance or similarity matrix

    n_jobs : int, optional, default: 1
        Number of processes for parallel computations.

        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs

    Returns
    -------
    D_snn : ndarray
        Secondary distance SNN matrix

    References
    ---------- 
    .. [1] R. Jarvis and E. A. Patrick, “Clustering using a similarity measure
           based on shared near neighbors,” IEEE Transactions on Computers,
           vol. 22, pp. 1025–1034, 1973.

    .. [2] Flexer, A., & Schnitzer, D. (2013). Can Shared Nearest Neighbors
           Reduce Hubness in High-Dimensional Spaces? 2013 IEEE 13th
           International Conference on Data Mining Workshops, 460–467.
           http://doi.org/10.1109/ICDMW.2013.101
    """
    io.check_distance_matrix_shape(D)
    io.check_valid_metric_parameter(metric)
    n = D.shape[0]
    if metric == 'distance':
        self_value = 0.
        sort_order = 1
        exclude = np.inf
        kth = k
    if metric == 'similarity':
        self_value = 1.
        sort_order = -1
        exclude = -np.inf
        kth = n - k
    distance = D.copy()
    np.fill_diagonal(distance, exclude)
    
    if n_jobs == -1:
        n_jobs = cpu_count()
    if n_jobs > 1:
        knn_ctype = RawArray(ctypes.c_bool, D.size)
        knn = np.frombuffer(knn_ctype, dtype=bool).reshape(D.shape)
        D_snn_ctype = RawArray(ctypes.c_double, D.size)
        D_snn = np.frombuffer(D_snn_ctype, dtype=np.float64).reshape(D.shape)
        with Pool(processes=n_jobs,
                  initializer=_snn_init,
                  initargs=(distance, knn, D_snn)) as pool:
            for _ in pool.imap(
                func=partial(_snn_my_hood, k=k, kth=kth, sort_order=sort_order),
                iterable=range(n)):
                pass
            for _ in pool.imap(
                func=partial(_snn_our_hood, k=k, metric=metric),
                iterable=range(n)):
                pass
    else:
        knn = np.zeros_like(distance, bool)
        # find nearest neighbors for each point
        for i in range(n):
            di = distance[i, :]
            nn = np.argpartition(di, kth=kth)[::sort_order]
            knn[i, nn[0:k]] = True
        D_snn = np.zeros_like(distance)
        for i in range(n):
            knn_i = knn[i, :]
            j_idx = slice(i+1, n)
    
            # using broadcasting
            Dij = np.sum(np.logical_and(knn_i, knn[j_idx, :]), 1)
            if metric == 'distance':
                D_snn[i, j_idx] = 1. - Dij / k
            else: # metric == 'similarity':
                D_snn[i, j_idx] = Dij / k

    D_snn += D_snn.T
    np.fill_diagonal(D_snn, self_value)
    return D_snn

#==============================================================================
# #============================================================================
# #                               SIMHUB IN
# #============================================================================
#==============================================================================

def _shi_init_knn(D_, knn_):
    global distance, knn
    distance = D_
    knn = knn_
    return

def _shi_hood(i, s, sort_order):
    di = distance[i, :]
    # TODO change to np.partition for PERF
    nn = np.argsort(di)[::sort_order]
    knn[i, nn[:s]] = True
    return

def _shi_init_simhub(knn_, train_ind_, I_n_, D_shi_):
    global knn, train_ind, I_n, D_shi
    knn = knn_
    train_ind = train_ind_
    I_n = I_n_
    D_shi = D_shi_
    return

def _shi_simhub_vect(i, s):
    # using vectorization and broadcasting
    x = np.logical_and(knn[i, :], knn[train_ind, :])
    D_shi[i, :] = np.sum(x * I_n, axis=1)
    return

def _shi_simhub(i, s, m):
    # use non-vectorized loops
    for j in range(m):
        x = np.logical_and(knn[i, :], knn[j, :])
        D_shi[i, j] = np.sum(x * I_n)
    return

def simhubIN(D:np.ndarray, train_ind:np.ndarray=None,
             test_ind:np.ndarray=None, s:int=50, return_distances:bool=True,
             n_jobs:int=1):
    """Calculate dissimilarity based on hubness-aware SNN distances [1]_.

    Parameters
    ----------
    D : ndarray
        The ``n x s`` distance, where ``n`` and ``s``
        are the dataset and sample size, respectively.

    train_ind : ndarray, optional, default: None
        The index array that determines, to which data points the columns in
        `D` correspond. Not required, if `D` is a quadratic all-against-all
        distance matrix.

    test_ind : ndarray, optional, default: None
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    s : int, optional, default: 50
        Neighborhood size. Can be optimized as to minimize hubness.

    return_distances : bool, optional, default: True
        If True, return distances (1 - similarities).
        Otherwise return similarities.

    n_jobs : int, optional, default: 1
        Number of processes for parallel computations.

        - `1`: Don't use multiprocessing.
        - `-1`: Use all CPUs

    Returns
    -------
    D_shi : ndarray
        Secondary distance (simhubIN) matrix.

    References
    ----------
    .. [1] Tomašev, N., Mladenić, D., Tomasev, N., & Mladenić, D. (2012).
           Hubness-aware shared neighbor distances for high-dimensional
           $$k$$ -nearest neighbor classification.
           Knowledge and Information Systems, 39(1), 89–122.
           http://doi.org/10.1007/s10115-012-0607-5
    """
    if train_ind is None:
        io.check_distance_matrix_shape(D)
    else:
        io.check_sample_shape_fits(D, train_ind)
    # Assuming distances in D
    self_value = 0.
    sort_order = 1
    exclude = np.inf
    distance = D.copy()
    n, m = distance.shape
    if test_ind is None:
        n_ind = range(n)
    else:
        n_ind = test_ind
    # Exclude self distances
    if train_ind is None:
        np.fill_diagonal(distance, exclude)
    else:
        for j, sample in enumerate(train_ind):
            distance[sample, j] = exclude

    if n_jobs == -1:
        n_jobs = cpu_count()
    if n_jobs > 1:
        knn_ctype = RawArray(ctypes.c_bool, D.size)
        knn = np.frombuffer(knn_ctype, dtype=bool).reshape(D.shape)
        with Pool(processes=n_jobs,
                  initializer=_shi_init_knn,
                  initargs=(distance, knn)) as pool:
            for _ in pool.imap(
                func=partial(_shi_hood, s=s, sort_order=sort_order),
                iterable=range(n)):
                pass
    else:
        knn = np.zeros_like(distance, bool)
        # find nearest neighbors for each point
        for i in range(n):
            di = distance[i, :]
            # TODO change to np.partition for PERF
            nn = np.argsort(di)[::sort_order]
            knn[i, nn[:s]] = True
    del distance

    # "Occurence informativeness"
    occ_inf_knn = knn[:m, :].copy()
    np.fill_diagonal(occ_inf_knn, True)
    N_s = occ_inf_knn.sum(axis=0)
    I_n = np.log(m / N_s)
    del occ_inf_knn

    # simhub calculation
    if train_ind is None:
        train_ind = ...
    if n_jobs > 1:
        D_shi_ctype = RawArray(ctypes.c_double, D.size)
        D_shi = np.frombuffer(D_shi_ctype, dtype=np.float64).reshape(D.shape)
        with Pool(processes=n_jobs,
                  initializer=_shi_init_simhub,
                  initargs=(knn, train_ind, I_n, D_shi)) as pool:
            if m < 2000:
                for _ in pool.imap(
                    func=partial(_shi_simhub_vect, s=s),
                    iterable=n_ind):
                    pass
            else:
                for _ in pool.imap(
                    func=partial(_shi_simhub, s=s, m=m),
                    iterable=n_ind):
                    pass
    else:
        D_shi = np.zeros_like(D)
        if m < 2000: # using vectorization and broadcasting
            for i in n_ind:
                x = np.logical_and(knn[i, :], knn[train_ind, :])
                D_shi[i, :] = np.sum(x * I_n, axis=1)
        else: # use non-vectorized loops
            for i in n_ind:
                for j in range(m):
                    x = np.logical_and(knn[i, :], knn[j, :])
                    D_shi[i, j] = np.sum(x * I_n)
    del knn
    # Normalization to [0, 1] range
    D_shi /= (s * np.log(m))

    # Convert to distances
    if return_distances:
        D_shi *= -1
        D_shi += 1
    else:
        self_value = 1

    if test_ind is None:
        # Ensure correct self distances and return sec. dist. matrix
        np.fill_diagonal(D_shi, self_value)
        return D_shi
    else:
        # only return test-train-distances (there are no self distances here)
        return D_shi[test_ind]

def simhub(D:np.ndarray, y:np.ndarray, train_ind:np.ndarray=None, 
           test_ind:np.ndarray=None, s:int=50, return_distances:bool=True,
           vect_usage:int=0):
    """Calculate dissimilarity based on hubness-aware SNN distances [1]_.

    Parameters
    ----------
    D : ndarray
        The ``n x s`` distance or similarity matrix, where ``n`` and ``s``
        are the dataset and sample size, respectively.

    y : ndarray or None
        Class labels. Required for supervised simhub (simhubIN + simhubPUR).
        If None, calculate unsupervised simhubIN as per equation (6) in [1]_.

    train_ind : ndarray, optional, default: None
        The index array that determines, to which data points the columns in
        `D` correspond. Not required, if `D` is a quadratic all-against-all
        distance matrix.

    test_ind : ndarray, optional, default: None
        Define data points to be hold out as part of a test set. Can be:

        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set.

    s : int, optional, default: 50
        Neighborhood size. Can be optimized as to minimize hubness.

    return_distances : bool, optional, default: True
        If True, return distances (1 - similarities).
        Otherwise return similarities.

    vect_usage : int, optional, default: 0
        If > 0, always use vectorization for the inner simhub loop.
        If < 0, always use nested loops.
        If == 0, this is dependent on data set size
        and vectorization is used if ``n >= 2000``.

    Returns
    -------
    D_shi : ndarray
        Secondary distance (simhubIN) matrix.

    References
    ----------
    .. [1] Tomašev, N., Mladenić, D.(2012).
           Hubness-aware shared neighbor distances for high-dimensional
           $$k$$ -nearest neighbor classification.
           Knowledge and Information Systems, 39(1), 89–122.
           http://doi.org/10.1007/s10115-012-0607-5
    """
    if train_ind is None:
        io.check_distance_matrix_shape(D)
    else:
        io.check_sample_shape_fits(D, train_ind)
    # Assuming distances in D
    self_value = 0.
    sort_order = 1
    exclude = np.inf
    distance = D.copy()
    n, m = distance.shape
    if not 0 < s < m:
        raise ValueError("Neighbor hood size s, must be [1, {}-1], but "
                         "was {}.".format(m, s))
    if test_ind is None:
        n_ind = range(n)
    else:
        n_ind = test_ind
    # Exclude self distances
    if train_ind is None:
        np.fill_diagonal(distance, exclude)
    else:
        for j, sample in enumerate(train_ind):
            distance[sample, j] = exclude

    knn = np.zeros_like(distance, bool)
    
    # find nearest neighbors for each point
    for i in range(n):
        di = distance[i, :]
        # TODO change to np.partition for PERF
        nn = np.argsort(di)[::sort_order]
        knn[i, nn[:s]] = True
    del distance

    # Reverse nearest neighbor count
    N_s = knn[:m, :].sum(axis=0)

    if y is not None:
        # Set of class labels
        C = np.unique(y)

        # Class specific reverse nearest neighbors
        N_sc = np.zeros((C.size, m))
        for c_idx, c_val in enumerate(C):
            N_sc[c_idx, :] = np.sum(knn[:m, :] * (y==c_val).reshape(-1, 1), axis=0)
        assert np.alltrue(N_sc.sum(axis=0) == N_s), "N_s,c(x) don't sum up to N_s(x)"

        # Account for each point being the 0th nearest neighbor
        N_sc += 1
    # In any case: the same for N_s
    N_s += 1

    if y is not None:
        # non-homogeneity (inconsistency) in occurrence
        N_sc /= N_s
        HR_s = -np.sum(N_sc * np.log(N_sc), axis=0)

        # Information gain
        max_H_s = np.log(C.size)
        info_gain = max_H_s - HR_s
    else: # set a dummy value for unsupervised mode
        info_gain = 1

    # "occurrence informativeness"
    I_n = np.log(m / N_s)

    # simhub calculation
    D_shi = np.zeros_like(D)
    if train_ind is None:
        train_ind = ...
    if vect_usage > 0 or (vect_usage == 0 and m < 2000):
        # using vectorization and broadcasting
        for i in n_ind:
            x = np.logical_and(knn[i, :], knn[train_ind, :])
            D_shi[i, :] = np.sum(x * I_n * info_gain, axis=1)
    else: # use non-vectorized loops
        for i in n_ind:
            for j in range(m):
                x = np.logical_and(knn[i, :], knn[j, :])
                D_shi[i, j] = np.sum(x * I_n * info_gain)
    del knn
    # Normalization to [0, 1] range
    if y is None:
        D_shi /= (s * np.log(m))
    else:
        D_shi /= (s * np.log(m) * max_H_s)

    # Convert to distances
    if return_distances:
        D_shi *= -1
        D_shi += 1
    else:
        self_value = 1

    if test_ind is None:
        # Ensure correct self distances and return sec. dist. matrix
        np.fill_diagonal(D_shi, self_value)
        return D_shi
    else:
        # only return test-train-distances (there are no self distances here)
        return D_shi[test_ind]

if __name__ == '__main__':
    from hub_toolbox.hubness import hubness
    from hub_toolbox.knn_classification import score
    D, y, X = io.load_dexter()
    print("D", D.shape)
    print("y", y.shape)
    print("X", X.shape)
    D_shi = simhub(D, y=None)
    D_snn = shared_nearest_neighbors(D, k=50)
    h = hubness(D_shi, k=5)
    h_snn = hubness(D_snn, k=5)
    acc = score(D_shi, y, 5)
    acc_snn = score(D_snn, y, 5)
    
    D_sh = simhub(D=D, y=y)
    h_sh = hubness(D_sh, k=5)
    acc_sh = score(D_sh, y, 5)
    print("hubness SNN:", h_snn[0])
    print("hubness SHI:", h[0])
    print("hubness SH :", h_sh[0])

    print("kNN SNN:", acc_snn[0][0, 0])
    print("kNN SHI:", acc[0][0, 0])
    print("kNN SH :", acc_sh[0][0, 0])
