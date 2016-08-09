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

import numpy as np
from scipy.sparse.base import issparse
from scipy.sparse.lil import lil_matrix
from hub_toolbox import Logging
    
def local_scaling(D:np.ndarray, k:int=7, metric:str='distance',
                  test_set_ind:np.ndarray=None):
    """Transform a distance matrix with Local Scaling.
    
    Transforms the given distance matrix into new one using local scaling [1]
    with the given neighborhood radius k. There are two types of local
    scaling methods implemented. The original one and NICDM, both reduce
    hubness in distance spaces, similarly to Mutual Proximity.
    
    Parameters:
    -----------
    D : ndarray or csr_matrix
        The n x n symmetric distance (similarity) matrix.
    
    k : int, optional (default: 7)
        Neighborhood radius for local scaling.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix.
        NOTE: self similarities in D_ls are set to np.inf
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    Returns:
    --------
    D_ls : ndarray
        Secondary distance LocalScaling matrix.
    
    See also:
    ---------
    [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
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
    D = np.copy(D)
    
    if metric == 'similarity':
        sort_order = -1
        exclude = -np.inf
        self_value = np.inf
        if issparse(D):
            log.warning("Sparse matrix support for LS is experimental.")
    else: # metric == 'distance':
        sort_order = 1
        exclude = np.inf
        self_value = 0
        if issparse(D):
            log.error("Sparse distance matrices are not supported.")
            raise NotImplementedError(
                      "Sparse distance matrices are not supported.") 
            
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
        tmp[0] = self_value
        tmp[1:] = D[i, i+1:] / np.sqrt(r[i] * r[i+1:])
        D_ls[i, i:] = tmp
    # copy triu to tril -> symmetric matrix (diag=zeros)
    # NOTE: does not affect self values, since inf+inf=inf and 0+0=0
    D_ls += D_ls.T
    
    if issparse(D):
        return D_ls.tocsr()
    else:
        return D_ls

def nicdm(D:np.ndarray, k:int=7, metric:str='distance', 
          test_set_ind:np.ndarray=None):
    """Transform a distance matrix with local scaling variant NICDM.
    
    Transforms the given distance matrix into new one using NICDM [1]
    with the given neighborhood radius k. There are two types of local
    scaling methods implemented. The original one and the non-iterative 
    contextual dissimilarity measure, both reduce hubness in distance spaces, 
    similarly to Mutual Proximity.
    
    Parameters:
    -----------
    D : ndarray
        The n x n symmetric distance (similarity) matrix.
    
    k : int, optional (default: 7)
        Neighborhood radius for local scaling.
    
    metric : {'distance', 'similarity'}, optional (default: 'distance')
        Define, whether matrix 'D' is a distance or similarity matrix
        
    test_sed_ind : ndarray, optional (default: None)
        Define data points to be hold out as part of a test set. Can be:
        - None : Rescale all distances
        - ndarray : Hold out points indexed in this array as test set. 
        
    Returns:
    --------
    D_nicdm : ndarray
        Secondary distance NICDM matrix.
    
    See also:
    ---------
    [1] Schnitzer, D., Flexer, A., Schedl, M., & Widmer, G. (2012). 
    Local and global scaling reduce hubs in space. The Journal of Machine 
    Learning Research, 13(1), 2871–2902.
    """

class LocalScaling():
    
    
    def __init__(self, D, k:int=7, scalingType='nicdm', isSimilarityMatrix=False):
        """Usage:
        ls = local_scaling(D, k, scalingType) 
            - Applies local scaling to the distance
             matrix D (NxN). The parameter k sets the neighborhood radius. 
        ls.perform_local_scaling()
            - Return the scaled distance matrix.
        
        Possible types (scalingType parameter):
          'original': Original Local Scaling using the distance of the k'th
             nearest neighbor.
          'nicdm': Local Scaling using the average distance of the k nearest
             neighbors.
        Create an instance for local scaling. 
        Parameters:
        k... neighborhood radius (DEFAULT = 7)
        scalingType... local scaling algorithm ['original', 'nicdm'] (DEFAULT='nicdm')
        """
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
        """Transform distance matrix using local scaling."""
        
        if self.scalingType == 'original':
            Dls = self.ls_k(test_set_mask)
        elif self.scalingType == 'nicdm':
            Dls = self.ls_nicdm(test_set_mask)
        else:
            self.warning("Invalid local scaling type!\n"+\
                         "Use: \nls = LocalScaling(D, 'original'|'nicdm')\n"+\
                         "Dls = ls.perform_local_scaling()")
            Dls = np.array([])
                
        return Dls
                
    def ls_k(self, test_set_mask=None):
        """Perform local scaling (original), using the k-th nearest neighbor."""
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)        
        
        length_D = np.max(np.shape(self.D))
        r = np.zeros((length_D, 1))
        for i in range(length_D):
            if issparse(self.D):
                di = self.D[i, train_set_mask].toarray()
            else:
                di = self.D[i, train_set_mask]
            di[i] = self.exclude
            nn = np.argsort(di)[::self.sort_order]
            r[i] = di[nn[self.k-1]] #largest similarities or smallest distances
        
        n = np.shape(self.D)[0]
        if issparse(self.D):
            Dls = dok_matrix(self.D.shape)
        else:
            Dls = np.zeros(np.shape(self.D), dtype = self.D.dtype)
        for i in range(n):
            for j in range(i+1, n):
                if self.isSimilarityMatrix:
                    #Dls[i, j] = np.exp(-self.D[i, j] / np.sqrt( r[i] * r[j] ))
                    Dls[i, j] = self.D[i, j] / np.sqrt( r[i] * r[j] )
                else:
                    Dls[i, j] = self.D[i, j] / np.sqrt( r[i] * r[j] )
                Dls[j, i] = Dls[i, j]
        if issparse(self.D):
            return Dls.tocsr()
        else:
            return Dls
        
    def ls_nicdm(self, test_set_mask=None):
        """Local scaling variant: Non-Iterative Contextual Dissimilarity Measure
            This uses the mean over the k nearest neighbors.
        """
         
        #=======================================================================
        # if self.isSimilarityMatrix:
        #     return self.ls_nicdm_sim(test_set_mask)
        # 
        #=======================================================================
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)
             
        length_D = np.max(np.shape(self.D))
        r = np.zeros((length_D, 1))
        np.fill_diagonal(self.D, np.inf)
        for i in range(length_D):
            di = self.D[i, :].copy()
            di[i] = self.exclude
            di = di[train_set_mask]
            nn = np.argsort(di)[::self.sort_order]
            r[i] = np.mean(di[nn[0:self.k]]) # largest sim. or smallest dist.
        rg = self.local_geomean(r)
         
        if self.isSimilarityMatrix:
            self.D = 1 - self.D / self.D.max()
        Dnicdm = np.zeros(np.shape(self.D), dtype = self.D.dtype)
        for i in range(length_D):
            for j in range(i+1, length_D):
                Dnicdm[i, j] = (rg * self.D[i, j]) / np.sqrt( r[i] * r[j] )
                Dnicdm[j, i] = Dnicdm[i, j]
        if self.isSimilarityMatrix:
            Dnicdm = 1 - Dnicdm / Dnicdm.max() 
         
        return Dnicdm
    
    #===========================================================================
    # def ls_nicdm_sim(self, test_set_mask=None):
    #     """Local scaling variant: Non-Iterative Contextual Dissimilarity Measure
    #         This uses the mean over the k nearest neighbors.
    #     """
    #      
    #     if test_set_mask is not None:
    #         train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
    #     else:
    #         train_set_mask = np.ones(self.D.shape[0], np.bool)
    #          
    #     if self.isSimilarityMatrix:
    #         self.D /= self.D.max()
    #         
    #     length_D = np.max(np.shape(self.D))
    #     f = np.zeros((length_D, 1))
    #     np.fill_diagonal(self.D, self.exclude)
    #     for i in range(length_D):
    #         si = self.D[i, train_set_mask]
    #         #di[i] = self.exclude
    #         nn = np.argsort(si)[::self.sort_order]
    #         f[i] = np.mean(1-si[nn[0:self.k]]) # largest sim. or smallest dist.
    # 
    #     fg = self.local_geomean(f)
    #      
    #     Snicdm = np.zeros(np.shape(self.D), dtype = self.D.dtype)
    #     for i in range(length_D):
    #         for j in range(i+1, length_D):
    #             Snicdm[i, j] = 1 - (fg**2 * (1-self.D[i, j])) / ( f[i] * f[j] )
    #             #Snicdm[i, j] = 1 - ((1-self.D[i, j]) * (f[i] * f[j])) / (fg**2) 
    #             Snicdm[j, i] = Snicdm[i, j]
    #     
    #     np.fill_diagonal(Snicdm, 1)
    #      
    #     return Snicdm
    #===========================================================================
            
    
        
