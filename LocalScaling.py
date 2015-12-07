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

class LocalScaling():
    """Transform a distance matrix with Local Scaling.
    
    """
    
    def __init__(self, D, k:int=7, scalingType='nicdm'):
        """ Create an instance for local scaling. 
        Parameters:
        k... neighborhood radius (DEFAULT = 7)
        scalingType... local scaling algorithm ['original', 'nicdm'] (DEFAULT='nicdm')
        """
        self.D = np.copy(D)
        self.k = k
        self.scalingType = scalingType
        
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
            di = self.D[i, train_set_mask]
            di[i] = np.inf
            nn = np.argsort(di)
            r[i] = di[nn[self.k-1]]
        
        n = np.shape(self.D)[0]
        Dls = np.zeros(np.shape(self.D), dtype = self.D.dtype)
        for i in range(n):
            for j in range(i+1, n):
                Dls[i, j] = self.D[i, j] / np.sqrt( r[i] * r[j] )
                Dls[j, i] = Dls[i, j]
        
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
            nn = np.argsort(di)
            r[i] = np.mean(di[nn[0:self.k]])
        rg = self.local_geomean(r)
         
        Dnicdm = np.zeros(np.shape(self.D), dtype = self.D.dtype)
        for i in range(length_D):
            for j in range(i+1, length_D):
                Dnicdm[i, j] = (rg * self.D[i, j]) / np.sqrt( r[i] * r[j] )
                Dnicdm[j, i] = Dnicdm[i, j]
         
        return Dnicdm
            
    def local_geomean(self, x):
        return np.exp(np.sum(np.log(x)) / np.max(np.shape(x)))
        
    
    def warning(self, *objs):
        print("WARNING: ", *objs, file=sys.stderr)
