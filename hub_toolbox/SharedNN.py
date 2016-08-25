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

def shared_nearest_neighbors(D:np.ndarray, k:int=10, metric='distance'):
    """Transform distance matrix using shared nearest neighbors [1]_.
    
    SNN similarity is based on computing the overlap between the k nearest 
    neighbors of two objects. SNN approaches try to symmetrize nearest neighbor 
    relations using only rank and not distance information [2]_.
    
    Parameters
    ----------
    D : np.ndarray
        The n x n symmetric distance (similarity) matrix.
        
    k : int, optional (default: 10)
        Neighborhood radius: The k nearest neighbors are used to calculate SNN.
        
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether the matrix 'D' is a distance or similarity matrix

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
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric == 'distance':
        self_value = 0.
        sort_order = 1
        exclude = np.inf
    elif metric == 'similarity':
        self_value = 1.
        sort_order = -1
        exclude = -np.inf
    else:
        raise ValueError("Parameter 'metric' must be "
                         "'distance' or 'similarity'.")
    
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

class SharedNN():
    """ DEPRECATED class."""
    def __init__(self, D, k=10, isSimilarityMatrix=False):
        """DEPRECATED"""
        print("DEPRECATED: Please use SharedNN.shared_nearest_neighbors() "
              "instead.", file=sys.stderr)
        self.D = np.copy(D)
        self.k = k
        if isSimilarityMatrix:
            self.sort_order = -1
        else:
            self.sort_order = 1
        
    def perform_snn(self):
        """Transform distance matrix using shared nearest neighbor."""
        if self.sort_order == -1:
            metric = 'similarity'
        else:
            metric = 'distance'
        return shared_nearest_neighbors(self.D, self.k, metric)
