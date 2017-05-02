#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the HUB TOOLBOX available at
http://ofai.at/research/impml/projects/hubology.html
Source code is available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2016-2017, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import unittest
import numpy as np
from scipy.spatial.distance import squareform
from hub_toolbox.Distances import euclidean_distance
from hub_toolbox.LocalScaling import local_scaling, nicdm
from hub_toolbox.Hubness import hubness
from hub_toolbox.KnnClassification import score

class TestLocalScaling(unittest.TestCase):
    """Unit tests for the LocalScaling class"""

    def setUpMod(self, mode='rnd'):
        np.random.seed(626)
        if mode == 'rnd':
            points = 200 # 200
            dim = 500 # 500
            self.vector = 99. * (np.random.rand(points, dim) - 0.5)
            self.label = np.random.randint(0, 5, points)
            self.dist = euclidean_distance(self.vector)
        elif mode == 'toy':
            # LS/NICDM ground truth calculated in spreadsheet for toy example
            self.dist = squareform([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9])
            self.ls_dist_truth = squareform(
                [0.486582881, 0.1535182751, 0.9816843611, 0.7364028619, 
                 0.6321205588, 0.6471339185, 0.9342714714, 0.9844961464, 
                 0.8646647168, 0.8150186001])
            self.nicdm_dist_truth = squareform(
                [0.2936782173, 0.1641711143, 0.7285259947, 0.4153237178, 
                 0.381499195, 0.3526961306, 0.5629896449, 0.7886525234, 
                 0.5395213357, 0.4489088861])
            self.vector = None
            self.label = None

    def tearDown(self):
        del self.dist, self.label, self.vector

    def test_local_scaling(self):
        self.setUpMod('toy')
        dist_calc = local_scaling(self.dist, k=2)
        calc_equals_truth = np.allclose(dist_calc, self.ls_dist_truth)
        return self.assertTrue(calc_equals_truth)
 
    def test_ls_basic_requirements(self):
        """Test that matrix is symmetric, diag==0, and in range [0, 1]"""
        self.setUpMod('rnd')
        ls_dist = local_scaling(self.dist)
        symmetric = np.all(ls_dist == ls_dist.T)
        diag_zero = np.all(ls_dist.diagonal() == 0.)
        correct_range = ls_dist.min() >= 0. and ls_dist.max() <= 1.
        return self.assertTrue(symmetric and diag_zero and correct_range)
 
    def test_ls_dist_equals_sim(self):
        """Test for equal RANKS using dist. vs. sim. (LS_dist != 1-LS_sim).
           Using hubness and k-NN accuracy as proxy."""
        self.setUpMod('rnd')
        ls_dist = local_scaling(self.dist, metric='distance')
        ls_sim = local_scaling(1 - self.dist, metric='similarity')
        h_dist, _, _ = hubness(ls_dist, metric='distance')
        h_sim, _, _ = hubness(ls_sim, metric='similarity')
        acc_dist, _, _ = score(ls_dist, self.label, metric='distance')
        acc_sim, _, _ = score(ls_sim, self.label, metric='similarity')
        dist_sim_equal_in_hubness_knn = np.allclose(h_dist, h_sim) and \
                                        np.allclose(acc_dist, acc_sim)
        return self.assertTrue(dist_sim_equal_in_hubness_knn)

    def test_ls_parallel_equals_sequential(self):
        self.setUpMod('rnd')
        ls_dist_par = local_scaling(self.dist, n_jobs=4)
        ls_dist_seq = local_scaling(self.dist, n_jobs=1)
        return np.testing.assert_array_equal(ls_dist_seq, ls_dist_par)

    def test_nicdm(self):
        self.setUpMod('toy')
        dist_calc = nicdm(self.dist, k=2)
        calc_equals_truth = np.allclose(dist_calc, self.nicdm_dist_truth)
        return self.assertTrue(calc_equals_truth)
 
    def test_nicdm_basic_requirements(self):
        """Test that matrix is symmetric, diag==0, and in range [0, inf)"""
        self.setUpMod('rnd')
        nicdm_dist = nicdm(self.dist)
        symmetric = np.all(nicdm_dist == nicdm_dist.T)
        diag_zero = np.all(nicdm_dist.diagonal() == 0.)
        correct_range = nicdm_dist.min() >= 0.
        return self.assertTrue(symmetric and diag_zero and correct_range)
 
    def test_nicdm_similarity_based(self):
        """There is no similarity-based NICDM"""
        self.setUpMod('toy')
        return self.assertRaises(NotImplementedError)

    def test_nicdm_parallel_equals_sequential(self):
        self.setUpMod('rnd')
        ls_dist_par = nicdm(self.dist, n_jobs=4)
        print(ls_dist_par)
        ls_dist_seq = nicdm(self.dist, n_jobs=1)
        print(ls_dist_seq)
        return np.testing.assert_array_equal(ls_dist_seq, ls_dist_par)

if __name__ == "__main__":
    unittest.main()
