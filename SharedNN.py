"""
Transforms the given distance matrix into new one using a shared nearest
neighbor transform with the given neighborhood radius k.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

Usage:
  Dsnn = shared_nn(D, k) - Use SNN with a neighborhood radius of k on the
     given distance matrix. The new distances are returned in Dsnn.

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-23
"""

import numpy as np

class SharedNN():
    """Transform a distance matrix with shared nearest neighbor.
    
    """
    
    def __init__(self, D, k = None):
        self.D = np.copy(D)
        if k is None:
            print("No neighborhood radius given. Using k=10")
            self.k = 10
        else:
            self.k = k
        
    def perform_snn(self):
        """Transform distance matrix using shared nearest neighbor."""
        
        n = np.shape(self.D)[0]
        z = np.zeros_like(self.D, bool)
        for i in range(n):
            di = self.D[i, :]
            di[i] = np.inf
            nn = np.argsort(di)
            z[i, nn[0:self.k]] = 1
            
        Dsnn = np.zeros_like(self.D)
        for i in range(n):
            zi = z[i, :]
            j_idx = np.arange(i+1, n)
            
            # numpy: automatic broadcasting instead of bsxfun()
            Dij = np.sum(np.logical_and(zi, z[j_idx, :]), 1)
            
            Dsnn[i, j_idx] = 1 - Dij / self.k
            Dsnn[j_idx, i] = Dsnn[i, j_idx]
    
        return Dsnn
