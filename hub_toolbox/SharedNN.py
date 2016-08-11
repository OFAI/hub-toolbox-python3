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
    """Transform distance matrix using shared nearest neighbors [1].
    
    SNN similarity is based on computing the overlap between the k nearest 
    neighbors of two objects. SNN approaches try to symmetrize nearest neighbor 
    relations using only rank and not distance information [2].
    
    Parameters:
    -----------
    D : np.ndarray
        The n x n symmetric distance (similarity) matrix.
        
    k : int, optional (default: 10)
        Neighborhood radius: The k nearest neighbors are used to calculate SNN.
        
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether the matrix 'D' is a distance or similarity matrix

    Returns:
    --------
    D_snn : ndarray
        Secondary distance SNN matrix
        
    See also:
    ---------    
    [1] R. Jarvis and E. A. Patrick, “Clustering using a similarity measure 
    based on shared near neighbors,” IEEE Transactions on Computers, 
    vol. 22, pp. 1025–1034, 1973.
    
    [2] Flexer, A., & Schnitzer, D. (2013). Can Shared Nearest Neighbors 
    Reduce Hubness in High-Dimensional Spaces? 2013 IEEE 13th International 
    Conference on Data Mining Workshops, 460–467. 
    http://doi.org/10.1109/ICDMW.2013.101
    """
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric == 'distance':
        sort_order = 1
    elif metric == 'similarity':
        sort_order = -1
    else:
        raise ValueError("Parameter 'metric' must be "
                         "'distance' or 'similarity'.")
    # need to copy matrix, because it is modified later
    self_D = D.copy()
    
    n = np.shape(self_D)[0]
    z = np.zeros_like(self_D, bool)
    for i in range(n):
        di = self_D[i, :]
        di[i] = np.inf
        nn = np.argsort(di)[::sort_order]
        z[i, nn[0:k]] = 1
    
    D_snn = np.zeros_like(self_D)
    for i in range(n):
        zi = z[i, :]
        j_idx = np.arange(i+1, n)
        
        # using broadcasting
        Dij = np.sum(np.logical_and(zi, z[j_idx, :]), 1)
        
        D_snn[i, j_idx] = 1 - Dij / k
        D_snn[j_idx, i] = D_snn[i, j_idx]

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
