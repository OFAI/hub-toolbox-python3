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

import numpy as np

class SharedNN():
    """
    Transforms the given distance matrix into new one using a shared nearest
    neighbor transform with the given neighborhood radius k.
    
    Usage:
    snn = SharedNN(D, k)
    D_snn = snn.perform_snn()
        - Use SNN with a neighborhood radius of k on the
        given distance matrix. The new distances are returned in D_snn.
    """
    
    def __init__(self, D, k=10, isSimilarityMatrix=False):
        self.D = np.copy(D)
        self.k = k
        if isSimilarityMatrix:
            self.sort_order = -1
        else:
            self.sort_order = 1
        
    def perform_snn(self):
        """Transform distance matrix using shared nearest neighbor."""
        
        n = np.shape(self.D)[0]
        z = np.zeros_like(self.D, bool)
        for i in range(n):
            di = self.D[i, :]
            di[i] = np.inf
            nn = np.argsort(di)[::self.sort_order]
            z[i, nn[0:self.k]] = 1
            
        Dsnn = np.zeros_like(self.D)
        for i in range(n):
            zi = z[i, :]
            j_idx = np.arange(i+1, n)
            
            # using broadcasting
            Dij = np.sum(np.logical_and(zi, z[j_idx, :]), 1)
            
            Dsnn[i, j_idx] = 1 - Dij / self.k
            Dsnn[j_idx, i] = Dsnn[i, j_idx]
    
        return Dsnn