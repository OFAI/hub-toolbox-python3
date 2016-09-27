#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2011-2016, Dominik Schnitzer, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""

try:
    import numpy
    import scipy
    del numpy
    del scipy
except ImportError:  # pragma: no cover
    raise ImportError("Could not import numpy and/or scipy.\n"
                      "Please make sure you install the following Python3 "
                      "packages: numpy, scipy and scikit-learn.\n"
                      "See the installation docs for more details:"
                      "http://hub-toolbox-python3.readthedocs.io/en/latest/"
                      "user/installation.html#numpy-scipy-scikit-learn")

try:
    import sklearn
    del sklearn
except ImportError:  # pragma: no cover
    print("Could not import scikit-learn. While most modules of the Hub "
          "Toolbox do not require it, it is still advised to install scikit-"
          "learn for Python3 to enable all functionality and unit tests.")

from . import Centering
from . import Distances
from . import GoodmanKruskal
from . import Hubness_parallel
from . import Hubness
from . import HubnessAnalysis
from . import IntrinsicDim
from . import IO
from . import KnnClassification
from . import LocalScaling
from . import Logging
from . import MutualProximity
from . import MutualProximity_parallel
from . import SharedNN
