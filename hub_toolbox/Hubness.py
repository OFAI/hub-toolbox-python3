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

import numpy as np
from scipy import stats as stat
from scipy import sparse
from hub_toolbox import IO, Logging
from scipy.sparse.base import issparse

class Hubness():
    """
    Computes the hubness of a distance matrix using its k nearest neighbors.
    Hubness [1] is the skewness of the n-occurrence histogram.
    
    [1] Hubs in Space: Popular Nearest Neighbors in High-Dimensional Data
    Radovanovic, Nanopoulos, Ivanovic, Journal of Machine Learning Research 2010
    """
    
    def __init__(self, D, k:int=5, isSimilarityMatrix:bool=False):
        self.log = Logging.ConsoleLogging()
        if isinstance(D, np.memmap):
            self.D = D
        else:
            self.D = IO.copy_D_or_load_memmap(D, writeable=False)
        self.k = k
        if isSimilarityMatrix:
            self.d_self = -np.inf
            self.sort_order = -1 # descending, interested in highest similarity
        else:
            self.d_self = np.inf
            self.sort_order = 1 # ascending, interested in smallest distance
        np.random.seed()
                
    def calculate_hubness(self, debug=False):
        """Calculate hubness."""
        
        if debug:
            self.log.message("Start hubness calculation "
                             "(skewness of {}-occurence)".format(self.k))
                            
        Dk = np.zeros( (self.k, np.size(self.D, 1)), dtype=np.float32 )
        
        if not isinstance(self.D, np.memmap) and \
            not sparse.issparse(self.D): 
            # correct self-distance must be ensured upstream for sparse/memmap
            # Set self dist to inf
            np.fill_diagonal(self.D, self.d_self)
            # make non-finite (NaN, Inf) appear at the end of the sorted list
            self.D[~np.isfinite(self.D)] = self.d_self
         
        
        for i in range(self.D.shape[0]):
            if debug and ((i+1)%10000==0 or i+1==self.D.shape[0]):
                self.log.message("NN: {} of {}.".
                                 format(i+1, self.D.shape[0]), flush=True)
            if issparse(self.D):
                d = self.D[i, :].toarray().ravel() # dense copy of one row
            elif isinstance(self.D, np.memmap):
                d = np.copy(d.astype(np.float)) # in memory copy
            else: # normal ndarray
                d = self.D[i, :]
            d[i] = self.d_self
            d[~np.isfinite(d)] = self.d_self
            # randomize the distance matrix rows to avoid the problem case
            # if all numbers to sort are the same, which would yield high
            # hubness, even if there is none
            rp = np.indices( (np.size(self.D, 1), ) )[0]
            rp = np.random.permutation(rp)
            d2 = d[rp]
            d2idx = np.argsort(d2, axis=0)[::self.sort_order]
            Dk[:, i] = rp[d2idx[0:self.k]]      
                   
        # N-occurence
        if debug:
            self.log.message("Counting n-occurence...")
        Nk = np.bincount(Dk.astype(int).ravel())    
        # Hubness
        if debug:
            self.log.message("Calculating hubness...")
        Sn = stat.skew(Nk)
         
        # return hubness, k-nearest neighbors, N occurence
        if debug:
            self.log.message("Hubness calculation done.", flush=True)
        return (Sn, Dk, Nk)