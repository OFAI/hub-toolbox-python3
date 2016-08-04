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

In the console (terminal application) change to the folder containing this file.

To build the package hub_toolbox:
python setup.py build

To install the package (with administrator rights):
sudo python setup.py install

"""

try:
    import setuptools
except ImportError:
    import warnings
    warnings.warn("setuptools not found, resorting to distutils"
                  #": unit test suite can not be simplenn_main from setup.py"
                  )
    setuptools = None

setup_options = {}

if setuptools is None:
    from distutils.core import setup  # @UnusedImport
else:
    from setuptools import setup  # @Reimport
    setup_options['test_suite'] = 'tests'
    
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
    requires=['numpy', 'scipy', 'sklearn'],
    packages=['hub_toolbox'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering"
    ],
    **setup_options
)
