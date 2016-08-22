#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2016, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import unittest
from scipy.spatial.distance import squareform
from hub_toolbox.Hubness import hubness

class TestHubness(unittest.TestCase):
    """Test hubness calculations"""

    def setUp(self):
        """Hubness truth: S_k=5, skewness calculated with bias"""
        self.dist = squareform([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9])
        self.hubness_truth = -0.2561204163

    def tearDown(self):
        del self.dist

    def test_hubness(self):
        """Test hubness against ground truth calc on spreadsheet"""
        Sk5, _, _ = hubness(self.dist, k=2)
        return self.assertAlmostEqual(Sk5, self.hubness_truth, places=10)

if __name__ == "__main__":
    unittest.main()
