#!/usr/bin/env python3
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

import sys
import numpy as np
from hub_toolbox.Hubness import hubness
from hub_toolbox import IO, Logging

def hubness(D:np.ndarray, k:int=5, metric='distance',
            verbose:int=0, n_jobs:int=-1):
    """
    .. note:: Deprecated in hub-toolbox 2.4: Parallel code merged into the
              Hubness module (hub_toolbox.Hubness). Please use that module
              instead.
              Hubness_parallel module will be removed in hub-toolbox 3.0.
    """
    return hubness(D=D, k=k, metric=metric, verbose=verbose, n_jobs=n_jobs)

class Hubness(): # pragma: no cover
    """
    .. note:: Deprecated in hub-toolbox 2.3
              Class will be removed in hub-toolbox 3.0.
              Please use static functions instead.
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

    def calculate_hubness(self, debug=False, n_jobs=-1):
        """Calculate hubness.

        .. note:: Deprecated in hub-toolbox 2.3
                  Class will be removed in hub-toolbox 3.0.
                  Please use static functions instead.
        """
        print("DEPRECATED: Please use Hubness_parallel.hubness().",
              file=sys.stderr)
        if self.sort_order == 1:
            metric = 'distance'
        elif self.sort_order == -1:
            metric = 'similarity'
        else:
            raise ValueError("sort_order must be -1 or 1.")

        return hubness(self.D, self.k, metric, debug, n_jobs)   

if __name__ == '__main__':
    """Simple test case"""
    from hub_toolbox.HubnessAnalysis import load_dexter
    D, l, v = load_dexter()
    Sn, Dk, Nk = hubness(D)
    print("Hubness =", Sn)
