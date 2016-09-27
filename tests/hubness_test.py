#!/usr/bin/env python3
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
import numpy as np
from scipy.spatial.distance import squareform
from hub_toolbox.Hubness import hubness
from hub_toolbox.Distances import euclidean_distance

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
        Sk5, _, _ = hubness(self.dist, k=2, verbose=1)
        return self.assertAlmostEqual(Sk5, self.hubness_truth, places=10)

    def test_hubness_return_values_are_self_consistent(self):
        """Test that the three returned values fit together"""
        np.random.seed(626)
        points = 200
        dim = 500
        vector = 99. * (np.random.rand(points, dim) - 0.5)
        dist = euclidean_distance(vector)
        k = 10
        Sk10, Dk10, Nk10 = hubness(dist, k=k)
        # Dk is just checked for correct shape
        correct_dim_Dk10 = Dk10.shape == (k, points)
        # Count k-occurence (different method than in module)
        Dk10 = Dk10.ravel()
        Nk10_true = np.zeros(points, dtype=int)
        for i in range(points):
            Nk10_true[i] = (Dk10 == i).sum()
        correct_Nk10 = np.all(Nk10 == Nk10_true)
        # Calculate skewness (different method than in module)
        x0 = Nk10 - Nk10.mean()
        s2 = (x0**2).mean()
        m3 = (x0**3).mean()
        s = m3 / (s2**1.5)
        Sk10_true = s
        correct_Sk10 = Sk10 == Sk10_true
        return self.assertTrue(correct_dim_Dk10 
                               and correct_Nk10 
                               and correct_Sk10)

if __name__ == "__main__":
    unittest.main()
