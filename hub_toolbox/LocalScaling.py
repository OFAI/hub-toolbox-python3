#!/usr/bin/env python
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

import sys
import numpy as np
from scipy.sparse.base import issparse
from scipy.sparse.lil import lil_matrix
from hub_toolbox import Logging
    
def local_scaling(D:np.ndarray, k:int=7, metric:str='distance',
                  test_set_ind:np.ndarray=None):
    """Transform a distance matrix with Local Scaling.
    
    Transforms the given distance matrix into new one using local scaling [1]_
    with the given k-th nearest neighbor. There are two types of local
    scaling methods implemented. The original one and NICDM, both reduce
    hubness in distance spaces, similarly to Mutual Proximity.
    
    Parameters
    ----------
    D : ndarray or csr_matrix
        The n x n symmetric distance (similarity) matrix.
    
    k : int, optional (default: 7)
        Neighborhood radius for local scaling.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix.
        
        NOTE: self similarities in sparse D_ls are set to np.inf
        
    test_sed_ind : ndarray, optional (default: None)
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
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")    
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
        if issparse(D):
            log.error("Sparse distance matrices are not supported.")
            raise NotImplementedError(
                "Sparse distance matrices are not supported.") 
            
    D = np.copy(D)
    n = D.shape[0]
    if test_set_ind is None:
        train_set_ind = slice(0, n) #take all        
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)
    
    r = np.zeros(n)
    for i in range(n):
        if issparse(D):
            di = D[i, train_set_ind].toarray()
        else:
            di = D[i, train_set_ind]
        di[i] = exclude
        nn = np.argsort(di)[::sort_order]
        r[i] = di[nn[k-1]] #largest similarities or smallest distances
    
    if issparse(D):
        D_ls = lil_matrix(D.shape)
    else:
        D_ls = np.zeros_like(D)
        
    for i in range(n):
        # vectorized inner loop: calc only triu part
        tmp = np.empty(n-i)
        tmp[0] = self_tmp_value
        if metric == 'similarity':
            tmp[1:] = np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
        else:
            tmp[1:] = 1 - np.exp(-1 * D[i, i+1:]**2 / (r[i] * r[i+1:]))
        D_ls[i, i:] = tmp
    # copy triu to tril -> symmetric matrix (diag=zeros)
    # NOTE: does not affect self values, since inf+inf=inf and 0+0=0
    D_ls += D_ls.T
    
    if issparse(D):
        return D_ls.tocsr()
    else:
        np.fill_diagonal(D_ls, self_value)
        return D_ls

def nicdm(D:np.ndarray, k:int=7, metric:str='distance', 
          test_set_ind:np.ndarray=None):
    """Transform a distance matrix with local scaling variant NICDM.
    
    Transforms the given distance matrix into new one using NICDM [1]_
    with the given neighborhood radius k (average). There are two types of 
    local scaling methods implemented. The original one and the non-iterative 
    contextual dissimilarity measure, both reduce hubness in distance spaces, 
    similarly to Mutual Proximity.
    
    Parameters
    ----------
    D : ndarray
        The n x n symmetric distance (similarity) matrix.
    
    k : int, optional (default: 7)
        Neighborhood radius for local scaling.
    
    metric : {'distance'}, optional (default: 'distance')
        Currently, only distance matrices are supported.
        
    test_sed_ind : ndarray, optional (default: None)
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
    if D.shape[0] != D.shape[1]:
        raise TypeError("Distance/similarity matrix is not quadratic.")
    if metric != 'similarity' and metric != 'distance':
        raise ValueError("Parameter 'metric' must be 'distance' "
                         "or 'similarity'.")
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
    
    if test_set_ind is None:
        train_set_ind = slice(0, n)
    else:
        train_set_ind = np.setdiff1d(np.arange(n), test_set_ind)

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
        D_nicdm[i, i+1:] = (r_geom * D[i, i+1:]) / np.sqrt(r[i] * r[i+1:])
    D_nicdm += D_nicdm.T
     
    return D_nicdm

def _local_geomean(x):
    return np.exp(np.sum(np.log(x)) / np.max(np.shape(x)))

##############################################################################
#
# DEPRECATED class
#
class LocalScaling():
    """
    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              Please use static functions instead.
    """
    
    def __init__(self, D, k:int=7, scalingType='nicdm', isSimilarityMatrix=False):
        """
        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        print("DEPRECATED: Please use LocalScaling.local_scaling() or "
              "LocalScaling.nicdm() instead.", file=sys.stderr)
        self.log = Logging.ConsoleLogging()
        self.D = np.copy(D)
        self.k = k
        self.scalingType = scalingType
        if isSimilarityMatrix:
            if scalingType=='nicdm':
                if issparse(D):
                    self.log.error("NICDM does not support sparse matrices.")
                    raise NotImplementedError(
                        "NICDM does not support sparse matrices.")
                else:
                    self.log.warning("NICDM does not support similarities. "
                        "Distances will be calculated as D=1-S/S.max and used "
                        "for NICDM scaling. Similarities are subsequently "
                        "obtained by the same procedure S=1-D/D.max")
            else:
                self.log.warning("Similarity-based LS support is experimental.")
        self.isSimilarityMatrix = isSimilarityMatrix
        if self.isSimilarityMatrix:
            self.sort_order = -1
            self.exclude = -np.inf 
        else:
            self.sort_order = 1
            self.exclude = np.inf
        if issparse(D):
            if isSimilarityMatrix:
                self.log.warning("Sparse matrix support for LS is experimental.")
            else:
                self.log.error("Sparse distance matrices are not supported.")
                raise NotImplementedError(
                               "Sparse distance matrices are not supported.")    
            
    def perform_local_scaling(self, test_set_mask=None):
        """
        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        if self.scalingType == 'original':
            Dls = self.ls_k(test_set_mask)
        elif self.scalingType == 'nicdm':
            Dls = self.ls_nicdm(test_set_mask)
        else:
            self.log.warning("Invalid local scaling type!\n"+\
                             "Use: \nls = LocalScaling(D, 'original'|'nicdm')\n"+\
                             "Dls = ls.perform_local_scaling()")
            Dls = np.array([])
        
        return Dls
    
    def ls_k(self, test_set_mask=None):
        """
        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance'
        return local_scaling(self.D, self.k, metric, test_set_mask)
    
    def ls_nicdm(self, test_set_mask=None):
        """
        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        if self.isSimilarityMatrix:
            metric = 'similarity'
        else:
            metric = 'distance'
        return nicdm(self.D, self.k, metric, test_set_mask)
