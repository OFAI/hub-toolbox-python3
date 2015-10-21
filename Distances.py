"""
Provides distances functions.

Created on Oct 21, 2015

@author: Roman Feldbauer
"""

import numpy as np

def cosine_distance(x):
    """Calculate the cosine distance."""
    xn = np.sqrt(np.sum(x**2, 1))
    x = x / np.tile(xn[:, np.newaxis], np.size(x, 1))
    D = 1 - np.dot(x, x.T )
    D[D<0] = 0
    D = np.triu(D, 0) + np.triu(D, 0).T
    return D