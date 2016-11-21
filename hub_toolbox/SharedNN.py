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

import sys
import numpy as np
from hub_toolbox import IO

def snn_sample(D:np.ndarray, k:int=10, metric='distance', 
               train_ind:np.ndarray=None, test_ind:np.ndarray=None):
    """Transform distance matrix using shared nearest neighbors [1]_.
    
    __DRAFT_VERSION__
    
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

    train_ind : ndarray, optional
        If given, use only these data points as neighbors for rescaling.

    test_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 

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
    IO._check_sample_shape_fits(D, train_ind)
    IO._check_valid_metric_parameter(metric)
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

    knn = np.zeros_like(distance, bool)
    
    # find nearest neighbors for each point
    for i in range(n):
        di = distance[i, :]
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


def shared_nearest_neighbors(D:np.ndarray, k:int=10, metric='distance'):
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
    IO._check_distance_matrix_shape(D)
    IO._check_valid_metric_parameter(metric)
    if metric == 'distance':
        self_value = 0.
        sort_order = 1
        exclude = np.inf
    if metric == 'similarity':
        self_value = 1.
        sort_order = -1
        exclude = -np.inf
    
    distance = D.copy()
    np.fill_diagonal(distance, exclude)
    n = np.shape(distance)[0]
    knn = np.zeros_like(distance, bool)
    
    # find nearest neighbors for each point
    for i in range(n):
        di = distance[i, :]
        nn = np.argsort(di)[::sort_order]
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

def simhubIN(D:np.ndarray, train_ind:np.ndarray=None, 
             test_ind:np.ndarray=None, s:int=50, return_distances:bool=True):
    """Calculate dissimilarity based on hubness-aware SNN distances [1]_.
    
    Parameters
    ----------
    D : ndarray
        The ``n x s`` distance or similarity matrix, where ``n`` and ``s``
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
        IO._check_distance_matrix_shape(D)
    else:
        IO._check_sample_shape_fits(D, train_ind)
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

    knn = np.zeros_like(distance, bool)
    
    # find nearest neighbors for each point
    for i in range(n):
        di = distance[i, :]
        nn = np.argsort(di)[::sort_order]
        knn[i, nn[:s]] = True
    del distance
    
    # "Occurence informativeness"
    occ_inf_knn = knn[:m, :].copy()
    #if train_ind is None:
    np.fill_diagonal(occ_inf_knn, True)
    """
    else:
        for j, sample in enumerate(train_ind):
            # treat x as its own 0th neighbor to avoid div/0
            occ_inf_knn[sample, j] = True
    """
    N_s = occ_inf_knn.sum(axis=0)
    I_n = np.log(m / N_s)
    del occ_inf_knn

    # simhub calculation
    D_shi = np.zeros_like(D)
    if train_ind is None:
        train_ind = ...
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

if __name__ == '__main__':
    from hub_toolbox.Hubness import hubness
    from hub_toolbox.KnnClassification import score
    D, y, X = IO.load_dexter()
    print("D", D.shape)
    print("y", y.shape)
    print("X", X.shape)
    D_shi = simhubIN(D)
    D_snn = shared_nearest_neighbors(D, k=50)
    h = hubness(D_shi, k=5)
    h_snn = hubness(D_snn, k=5)
    acc = score(D_shi, y, 5)
    acc_snn = score(D_snn, y, 5)
    print("hubness SHI:", h[0])
    print("hubness SNN:", h_snn[0])
    print("kNN SHI:", acc[0][0, 0])
    print("kNN SNN:", acc_snn[0][0, 0])

class SharedNN(): # pragma: no cover
    """
    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              Please use static functions instead.
    """
    def __init__(self, D, k=10, isSimilarityMatrix=False):
        """
        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        print("DEPRECATED: Please use SharedNN.shared_nearest_neighbors() "
              "instead.", file=sys.stderr)
        self.D = np.copy(D)
        self.k = k
        if isSimilarityMatrix:
            self.sort_order = -1
        else:
            self.sort_order = 1
        
    def perform_snn(self):
        """Transform distance matrix using shared nearest neighbor.

        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        if self.sort_order == -1:
            metric = 'similarity'
        else:
            metric = 'distance'
        return shared_nearest_neighbors(self.D, self.k, metric)
