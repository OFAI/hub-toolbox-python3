"""
Provides distance functions.

Created on Oct 21, 2015

@author: Roman Feldbauer
"""

import numpy as np
from scipy.spatial.distance import cdist
from enum import Enum

def cosine_distance(X):
    """Calculate the cosine distance between all pairs of vectors in X."""
    xn = np.sqrt(np.sum(X**2, 1))
    X = X / np.tile(xn[:, np.newaxis], np.size(X, 1))
    D = 1 - np.dot(X, X.T )
    D[D<0] = 0
    D = np.triu(D, 0) + np.triu(D, 0).T
    return D

def euclidean_distance(X):
    """Calculate the euclidean distances between all pairs of vectors in X."""
    #===========================================================================
    # # TODO vectorize to gain speed
    # D = np.zeros((X.shape[0], X.shape[0]))
    # for i in range(len(X)):
    #     for j in range(i+1, len(X)):
    #         D[i, j] = np.linalg.norm(X[i]-X[j])
    # D[D<0] = 0  # distances must not be negative
    # D += D.T    # set symmetric distances (dist(x_j, x_i) = dist(x_i, x_j))
    #===========================================================================
    # using scipy
    D = cdist(X, X, 'euclidean')
    return D

class Distance(Enum):
    """Enum for distance metrics."""
    
    cosine = 'cosine'
    euclidean = 'euclidean'
    skl = 'skl'   