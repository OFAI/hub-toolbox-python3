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
import numpy as np
from scipy.spatial.distance import pdist, squareform
from hub_toolbox.Distances import cosine_distance, euclidean_distance

class TestDistances(unittest.TestCase):

    def setUp(self):
        np.random.seed(626)
        self.vectors = 99. * (np.random.rand(400, 200) - 0.5)

    def tearDown(self):
        del self.vectors

    def test_cosine_dist_equal_to_scipy_pdist_cos(self):
        cos_dist = cosine_distance(self.vectors)
        cos_dist_scipy = squareform(pdist(self.vectors, 'cosine'))
        result = np.allclose(cos_dist, cos_dist_scipy)
        return self.assertTrue(result)
    
    def test_euclidean_dist_equal_to_scipy_pdist_eucl(self):
        eucl_dist = euclidean_distance(self.vectors)
        eucl_dist_scipy = squareform(pdist(self.vectors, 'euclidean'))
        result = np.allclose(eucl_dist, eucl_dist_scipy)
        return self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()