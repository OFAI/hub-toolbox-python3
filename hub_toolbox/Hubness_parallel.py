#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2017, Dominik Schnitzer and Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

import numpy as np
from hub_toolbox.Hubness import hubness as hubness_s

def hubness(D:np.ndarray, k:int=5, metric='distance',
            verbose:int=0, n_jobs:int=-1):
    """
    .. note:: Deprecated in hub-toolbox 2.4: Parallel code merged into the
              Hubness module (hub_toolbox.Hubness). Please use that module
              instead.
              Hubness_parallel module will be removed in hub-toolbox 3.0.
    """
    return hubness_s(D=D, k=k, metric=metric, verbose=verbose, n_jobs=n_jobs)

if __name__ == '__main__':
    """Simple test case"""
    from hub_toolbox.IO import load_dexter
    D, l, v = load_dexter()
    Sn, Dk, Nk = hubness(D)
    print("Hubness =", Sn)
