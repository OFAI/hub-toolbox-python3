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

from enum import Enum
import numpy as np
from scipy.spatial.distance import cdist

def cosine_distance(X):
    """Calculate the cosine distance between all pairs of vectors in `X`."""
    xn = np.sqrt(np.sum(X**2, 1))
    X = X / np.tile(xn[:, np.newaxis], np.size(X, 1))
    D = 1 - np.dot(X, X.T)
    D[D < 0] = 0
    D = np.triu(D, 1) + np.triu(D, 1).T
    return D

def euclidean_distance(X):
    """Calculate the euclidean distances between all pairs of vectors in `X`."""
    D = cdist(X, X, 'euclidean')
    return D

class Distance(Enum):
    """Enum for distance metrics.

    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              All functions now take str parameters directly.
    """
    cosine = 'cosine'
    euclidean = 'euclidean'
    skl = 'skl'
