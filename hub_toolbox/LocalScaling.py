"""
Transforms the given distance matrix into new one using local scaling [1]
with the given neighborhood radius k. There are two types of local
scaling methods implemented. The original one and NICDM, both reduce
hubness in distance spaces, similarly to Mutual Proximity.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

Usage:
  Dls = local_scaling(D, k, type) - Applies local scaling to the distance
     matrix D (NxN). The parameter k sets the neighborhood radius. type
     the scaling type. The scaled distance matrix is returned.

Possible types (type parameter):
  'original': Original Local Scaling using the distance of the k'th
     nearest neighbor.
  'nicdm': Local Scaling using the average distance of the k nearest
     neighbors.


[1] Local and global scaling reduce hubs in space, 
Schnitzer, Flexer, Schedl, Widmer, Journal of Machine Learning Research 2012

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-24
"""

import numpy as np
import sys
from scipy.sparse.base import issparse
from scipy.sparse.dok import dok_matrix
from hub_toolbox import Logging

class LocalScaling():
    """Transform a distance matrix with Local Scaling.
    
    """
    
    def __init__(self, D, k:int=7, scalingType='nicdm', isSimilarityMatrix=False):
        """ Create an instance for local scaling. 
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
         
        if test_set_mask is not None:
            train_set_mask = np.setdiff1d(np.arange(self.D.shape[0]), test_set_mask)
        else:
            train_set_mask = np.ones(self.D.shape[0], np.bool)
             
        length_D = np.max(np.shape(self.D))
        r = np.zeros((length_D, 1))
        np.fill_diagonal(self.D, np.inf)
        for i in range(length_D):
            di = self.D[i, train_set_mask]
            di[i] = self.exclude
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
            
    def local_geomean(self, x):
        return np.exp(np.sum(np.log(x)) / np.max(np.shape(x)))
        
    
    def warning(self, *objs):
        print("WARNING: ", *objs, file=sys.stderr)
