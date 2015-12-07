"""
Computes the hubness of a distance matrix using its k nearest neighbors.
Hubness [1] is the skewness of the n-occurrence histogram.

This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
(c) 2013, Dominik Schnitzer <dominik.schnitzer@ofai.at>

Usage:
  Sn = hubness(D) - Computes the hubness (Sk) of the n=5 occurrence histogram
     (standard)

  [Sn, Dk, Nk] hubness(D, k) - Computes the hubness of the n-occurrence
     histogram where n (k) is given. Nk is the n-occurrence histogram, Dk
     are the k nearest neighbors.

[1] Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data
Radovanovic, Nanopoulos, Ivanovic, Journal of Machine Learning Research 2010

This file was ported from MATLAB(R) code to Python3
by Roman Feldbauer <roman.feldbauer@ofai.at>

@author: Roman Feldbauer
@date: 2015-09-17
"""

import numpy as np
from scipy import stats as stat

class Hubness():
    """Computes the hubness of a distance matrix using its k nearest neighbors.
    
    Hubness is the skewness of the n-occurrence histogram.
    """
    
    def __init__(self, D, k: int = 5, isSimilarityMatrix: bool = False):
        self.D = np.copy(D)
        self.k = k
        if isSimilarityMatrix:
            self.d_self = -np.inf
            self.sort_order = -1 # descending, interested in highest similarity
        else:
            self.d_self = np.inf
            self.sort_order = 1 # ascending, interested in smallest distance
        np.random.seed()
                
    def calculate_hubness(self, debug = False):
        """Calculate hubness."""
        
        if debug:
            print("Hubness...")
                
        Dk = np.zeros( (self.k, np.size(self.D, 1)) )
        
        # Set self dist to inf
        np.fill_diagonal(self.D, self.d_self)
        # make non-finite (NaN, Inf) appear at the end of the sorted list
        self.D[~np.isfinite(self.D)] = self.d_self
            
        i = 0
        for d in self.D:
            if debug and (i % 1000 == 0):
                print("NN: {} of {}.".format(i, self.D.shape[0]))
            # randomize the distance matrix rows to avoid the problem case
            # if all numbers to sort are the same, which would yield high
            # hubness, even if there is none
            rp = np.indices( (np.size(self.D, 1), ) )[0]
            rp = np.random.permutation(rp)
            d2 = d[rp]
            d2idx = np.argsort(d2, axis=0)[::self.sort_order]
            Dk[:, i] = rp[d2idx[0:self.k]]      
            i += 1
            
        # N-occurence
        if debug:
            print("Counting n-occurence...")
        Nk = np.bincount(Dk.astype(int).ravel())    
        # Hubness
        if debug:
            print("Calculation skewness = hubness...")
        Sn = stat.skew(Nk)
         
        # return hubness, k-nearest neighbors, N occurence
        if debug:
            print("Hubness: done.")
        return (Sn, Dk, Nk)