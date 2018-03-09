#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2016-2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import unittest
import numpy as np
from hub_toolbox.distances import euclidean_distance
from hub_toolbox.global_scaling import mutual_proximity_empiric,\
    mutual_proximity_gaussi, mutual_proximity_gammai
from scipy.sparse.csr import csr_matrix
from scipy.spatial.distance import squareform

class TestMutualProximity(unittest.TestCase):
    """Unit tests for the MutualProximity class (serial computing)"""

    def setUpMod(self, mode='rnd'):
        np.random.seed(626)
        if mode == 'rnd':
            points = 50
            dim = 500
            self.vector = 99. * (np.random.rand(points, dim) - 0.5)
            self.label = np.random.randint(0, 5, points)
            self.dist = euclidean_distance(self.vector)
            # scale to [0, 1), avoiding 1: otherwise sparseMP != denseMP (by design)
            self.dist /= (self.dist.max() + 1e-12)
        elif mode == 'toy':
            # MP empiric ground truth calculated by hand for this toy example
            self.dist = squareform([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9])

            # MP with div/(n-0)
            self.mp_dist_truth = squareform([.6, .4, 1., .8, .6,
                                             .8, 1., 1., .8, 1.])
            """
            # MP with div/(n-1)
            self.mp_dist_truth = squareform([.5, .25, 1., .75, .5,
                                             .75, 1., 1., .75, 1.])

            # MP with div/(n-2)
            self.mp_dist_truth = squareform([1/3, 0., 1., 2/3, 1/3,
                                             2/3, 1., 1., 2/3, 1.])
            """
            self.vector = None
            self.label = None

    def tearDown(self):
        del self.dist, self.label, self.vector

    def test_mp_empiric_sample(self):
        """Test MP Emp Sample equals MP Emp when sample == population"""
        self.setUpMod('toy')
        mp_dist = mutual_proximity_empiric(self.dist, 'distance')
        y = np.array([0, 1, 2, 3, 4])
        mp_sample_dist = mutual_proximity_empiric(D=self.dist,
                                                  sample_ind=y,
                                                  metric='distance')
        return np.testing.assert_array_almost_equal(
            mp_dist, mp_sample_dist, decimal=7)

    """
    def test_mp_gaussi_sample(self):
        """'''Test MP Gaussi Sample.'''"""
        self.setUpMod('toy')
        mp_dist = mutual_proximity_gaussi(self.dist)
        y = np.array([0, 1, 2, 3, 4])
        mp_sample_dist = mutual_proximity_gaussi(self.dist[:, y], idx=y)
        mp_sample_equal_pop = np.alltrue(mp_dist == mp_sample_dist)
        #=======================================================================
        # print(self.dist)
        # print(mp_dist)
        # print(mp_sample_dist)
        #=======================================================================
        print("SampleMP-Gaussi with all pts equals MP-Gaussi:", mp_sample_equal_pop)
        y2 = np.array([1, 2, 4])
        mp_sample_dist2 = mutual_proximity_gaussi(self.dist[:, y2], idx=y2)
        print(self.dist[:, y2])
        print(mp_dist[:, y2])
        print(mp_sample_dist2)
        print(mp_sample_dist)
        #return self.assertTrue(mp_sample_equal_pop)
        return self.fail()
    """

    def test_mp_empiric(self):
        """Test MP Empiric for toy example (ground truth calc by hand)"""
        self.setUpMod('toy')
        mp_dist_calc = mutual_proximity_empiric(self.dist, 'distance', verbose=1)
        return np.testing.assert_array_almost_equal(
            mp_dist_calc, self.mp_dist_truth, decimal=7)

    def test_mp_empiric_all_zero_self_distances(self):
        self.setUpMod('rnd')
        mp_dist_calc = mutual_proximity_empiric(self.dist)
        mp_self_distances_all_zero = np.all(mp_dist_calc.diagonal() == 0.)
        return self.assertTrue(mp_self_distances_all_zero)

    def test_mp_empiric_symmetric(self):
        self.setUpMod('rnd')
        mp_dist = mutual_proximity_empiric(self.dist)
        return np.testing.assert_array_almost_equal(
            mp_dist, mp_dist.T, decimal=14)

    def test_mp_empiric_dist_equal_sim(self):
        self.setUpMod('rnd')
        sim = 1. - self.dist
        mp_dist = mutual_proximity_empiric(self.dist, 'distance')
        mp_sim = mutual_proximity_empiric(sim, 'similarity')
        return np.testing.assert_array_almost_equal(
            mp_dist, 1. - mp_sim, decimal=7)

    def test_mp_empiric_sparse_equal_dense(self):
        self.setUpMod('rnd')
        sim_dense = 1. - self.dist
        sim_sparse = csr_matrix(sim_dense)
        mp_dense = mutual_proximity_empiric(sim_dense, 'similarity')
        mp_sparse = mutual_proximity_empiric(
            sim_sparse, 'similarity', verbose=1, n_jobs=4)
        return np.testing.assert_array_almost_equal(
            mp_dense, mp_sparse.toarray(), decimal=7)

    def test_mp_gaussi(self):
        """Test MP GaussI for toy example (ground truth calc by 'hand')"""
        self.setUpMod('toy')
        mp_gaussi = mutual_proximity_gaussi(self.dist, verbose=1)
        # Calculated with formula (3) in JMLR paper, aided by LibreOffice Calc
        mp_gaussi_hand = np.array(
            [[0.155334048, 0.3466121867, 0.2534339319, 0.971773078, 0.575452874], 
             [0.3466121867, 0.0267023937, 0.4637020361, 0.6708772779, 0.9702788336], 
             [0.2534339319, 0.4637020361, 0.1354428205, 0.9899969991, 0.7660250185], 
             [0.971773078, 0.6708772779, 0.9899969991, 1.90126724466388e-05, 0.975462801], 
             [0.575452874, 0.9702788336, 0.7660250185, 0.975462801, 0.0003114667]])
        # Gaussians can go below distance 0; self dist anyway defined as 0.
        np.fill_diagonal(mp_gaussi_hand, 0.)
        return np.testing.assert_array_almost_equal(
            mp_gaussi, mp_gaussi_hand, decimal=7)

    def test_mp_gaussi_all_zero_self_distances(self):
        self.setUpMod('rnd')
        mp_dist = mutual_proximity_gaussi(self.dist)
        mp_self_dist_all_zero = np.all(mp_dist.diagonal() == 0.)
        return self.assertTrue(mp_self_dist_all_zero)

    def test_mp_gaussi_symmetric(self):
        self.setUpMod('rnd')
        mp_dist = mutual_proximity_gaussi(self.dist)
        return np.testing.assert_array_almost_equal(
            mp_dist, mp_dist.T, decimal=7)
    
    def test_mp_gaussi_dist_equal_sim(self):
        self.setUpMod('rnd')
        sim = 1. - self.dist
        mp_dist = mutual_proximity_gaussi(self.dist, 'distance')
        mp_sim = mutual_proximity_gaussi(sim, 'similarity')
        return np.testing.assert_array_almost_equal(
            mp_dist, 1. - mp_sim, decimal=7)

    def test_mp_gaussi_sparse_equal_dense(self):
        self.setUpMod('rnd')
        sim_dense = 1. - self.dist
        sim_sparse = csr_matrix(sim_dense)
        mp_dense = mutual_proximity_gaussi(sim_dense, 'similarity')
        mp_sparse = mutual_proximity_gaussi(sim_sparse, 'similarity')
        return np.testing.assert_array_almost_equal(
            mp_dense, mp_sparse.toarray(), decimal=7)

    def test_mp_gammai(self):
        """Test MP GammaI for toy example (ground truth calc by 'hand')"""
        self.setUpMod('toy')
        mp_gammai = mutual_proximity_gammai(self.dist, verbose=1)
        # Calculated with formula (3) in JMLR paper, aided by LibreOffice Calc
        mp_gammai_hand = np.array(
            [[0., 0.4334769987, 0.230927083, 0.9558409888, 0.6744697939],
             [0.4334769987, 0., 0.5761291218, 0.7088478962, 0.9585297208],
             [0.230927083, 0.5761291218, 0., 0.9817785746, 0.8286910286],
             [0.9558409888, 0.7088478962, 0.9817785746, 0., 0.9646050169],
             [0.6744697939, 0.9585297208, 0.8286910286, 0.9646050169, 0.]])
        return np.testing.assert_array_almost_equal(
            mp_gammai, mp_gammai_hand, decimal=7)

    def test_mp_gammai_all_zero_self_distances(self):
        self.setUpMod('rnd')
        mp_dist = mutual_proximity_gammai(self.dist)
        mp_self_dist_all_zero = np.all(mp_dist.diagonal() == 0.)
        return self.assertTrue(mp_self_dist_all_zero)

    def test_mp_gammai_symmetric(self):
        self.setUpMod('rnd')
        mp_dist = mutual_proximity_gammai(self.dist)
        return np.testing.assert_array_almost_equal(
            mp_dist, mp_dist.T, decimal=7)

    def test_mp_gammai_dist_equal_sim(self):
        self.setUpMod('rnd')
        #=====================================================================
        # sim = 1. - self.dist
        # mp_dist = mutual_proximity_gammai(self.dist, 'distance')
        # mp_sim = mutual_proximity_gammai(sim, 'similarity')
        # dist_allclose_one_minus_sim = np.allclose(mp_dist, 1. - mp_sim)
        #=====================================================================
        msg = "MP GammaI similarity differs from GammaI distance. "\
            + "Whether the currently implemented similarity function makes "\
            + "any sense, is yet to be investigated."
        return self.skipTest(msg)
        #return self.assertTrue(dist_allclose_one_minus_sim)

    def test_mp_gammai_sparse_equal_dense(self):
        self.setUpMod('rnd')
        sim_dense = 1. - self.dist
        sim_sparse = csr_matrix(sim_dense)
        mp_dense = mutual_proximity_gammai(sim_dense, 'similarity')
        mp_sparse = mutual_proximity_gammai(sim_sparse, 'similarity')
        dense_allclose_sparse = np.allclose(mp_dense, mp_sparse.toarray())
        return self.assertTrue(dense_allclose_sparse)

if __name__ == "__main__":
    unittest.main()
