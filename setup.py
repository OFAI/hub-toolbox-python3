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


Installation:
-------------
In the console (terminal application) change to the folder containing this file.

To build the package hub_toolbox:
python3 setup.py build

To install the package (with administrator rights):
sudo python3 setup.py install

To test the installation:
sudo python3 setup.py test

If this succeeds with an 'OK' message, you are ready to go.
Otherwise you may consider filing a bug report on github.
(Some skipped tests are perfectly fine, though.)
"""
import sys
if sys.version_info < (3, 4):
    sys.stdout.write("The HUB TOOLBOX requires Python 3.4\n"
                     "Please try to run as python3 setup.py or\n"
                     "update your Python environment.\n"
                     "Consider using Anaconda for easy package handling.\n")
    sys.exit(1)

try:
    import numpy, scipy, sklearn  # @UnusedImport
except ImportError:
    sys.stdout.write("The HUB TOOLBOX requires numpy, scipy and scikit-learn. "
                     "Please make sure these packages are available locally. "
                     "Consider using Anaconda for easy package handling.\n")

setup_options = {}

try:
    from setuptools import setup
    setup_options['test_suite'] = 'tests'
except ImportError:
    from distutils.core import setup
    import warnings
    warnings.warn("setuptools not found, resorting to distutils. "
                  "Unit tests won't be discovered automatically.")

setup(
    name = "hub_toolbox",
    version = "2.3",
    author = "Roman Feldbauer",
    author_email = "roman.feldbauer@ofai.at",
    maintainer = "Roman Feldbauer",
    maintainer_email = "roman.feldbauer@ofai.at",
    description = "Hubness reduction and analysis tools",
    license = "GNU GPLv3",
    keywords = ["machine learning", "data science"],
    url = "https://github.com/OFAI/hub-toolbox-python3",
    packages=['hub_toolbox', 'tests'],
    package_data={'hub_toolbox': ['example_datasets/*']},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    **setup_options
)
