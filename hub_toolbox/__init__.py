#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2018, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

__version__ = '3.0.dev1'

try:
    import numpy
    import scipy
    import sklearn
    del numpy
    del scipy
    del sklearn
except ImportError:  # pragma: no cover
    raise ImportError("Could not import numpy and/or scipy.\n"
                      "Please make sure you install the following Python3 "
                      "packages: numpy, scipy and scikit-learn.\n"
                      "See the installation docs for more details:"
                      "http://hub-toolbox-python3.readthedocs.io/en/latest/"
                      "user/installation.html#numpy-scipy-scikit-learn")

from hub_toolbox import centering
from hub_toolbox import distances
from hub_toolbox import goodman_kruskal
from . import Hubness_parallel
from hub_toolbox import hubness
from hub_toolbox import hubness_analysis
from hub_toolbox import intrinsic_dimension
from hub_toolbox import io
from hub_toolbox import knn_classification
from hub_toolbox import local_scaling
from hub_toolbox import logging
from hub_toolbox import global_scaling
from . import MutualProximity_parallel
from hub_toolbox import shared_neighbors
