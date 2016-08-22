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
from scipy.spatial.distance import squareform
from hub_toolbox.Distances import euclidean_distance
from hub_toolbox.SharedNN import shared_nearest_neighbors

class TestSharedNN(unittest.TestCase):

    def setUpMod(self, mode='rnd'):
        np.random.seed(626)
        if mode == 'rnd':
            points = 200
            dim = 500
            self.vector = 99. * (np.random.rand(points, dim) - 0.5)
            self.label = np.random.randint(0, 5, points)
            self.dist = euclidean_distance(self.vector)
            #self.dist /= (self.dist.max() + 1e-12)
        elif mode == 'toy':
            # SNN (k=2) ground truth calculated by hand for this toy example
            self.dist = squareform([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9])
            self.snn_dist_truth = squareform([.5, .5, .5, .5, .5, 
                                              .5, 0., 0., .5, .5])
            self.vector = None
            self.label = None
            
    def tearDown(self):
        del self.dist, self.label, self.vector

    def test_snn_matrix_basic_requirements(self):
        """Test that matrix is symmetric, diag==0, and in range [0, 1]"""
        self.setUpMod('rnd')
        snn_dist = shared_nearest_neighbors(self.dist)
        symmetric = np.all(snn_dist == snn_dist.T)
        diag_zero = np.all(snn_dist.diagonal() == 0.)
        correct_range = snn_dist.min() >= 0. and snn_dist.max() <= 1.
        return self.assertTrue(symmetric and diag_zero and correct_range)

    def test_snn(self):
        """Test correctness of SNN in toy example (hand-calculated)"""
        self.setUpMod('toy')
        snn_dist = shared_nearest_neighbors(self.dist, k=2)
        snn_calc_equal_truth = np.all(snn_dist == self.snn_dist_truth)
        return self.assertTrue(snn_calc_equal_truth)

    def test_snn_dist_equals_sim(self):
        """Test that SNN results are equivalent using distances or simil."""
        self.setUpMod('rnd')
        snn_dist = shared_nearest_neighbors(self.dist, metric='distance')
        snn_sim = shared_nearest_neighbors(1. - self.dist, metric='similarity')
        dist_equals_sim = np.allclose(snn_sim, 1.-snn_dist)
        return self.assertTrue(dist_equals_sim)

if __name__ == "__main__":
    unittest.main()
