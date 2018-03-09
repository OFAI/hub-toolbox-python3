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
from hub_toolbox.distances import euclidean_distance
from hub_toolbox.global_scaling import mutual_proximity_empiric as mpe_s
from hub_toolbox.global_scaling import mutual_proximity_gaussi as mpmvni_s
from hub_toolbox.global_scaling import mutual_proximity_gammai as mpgam_s 
from hub_toolbox.MutualProximity_parallel import mutual_proximity_empiric, \
    mutual_proximity_gaussi, mutual_proximity_gammai
from scipy.sparse.csr import csr_matrix

class TestMutualProximityParallel(unittest.TestCase):
    """Unit tests for MutualProximity_parallel class"""

    def setUp(self):
        np.random.seed(626)
        points = 100
        dim = 1000
        self.vector = 99. * (np.random.rand(points, dim) - 0.5)
        self.label = np.random.randint(0, 5, points)
        self.dist = euclidean_distance(self.vector)
        self.dist /= self.dist.max()

    def tearDown(self):
        del self.dist, self.label, self.vector

    def test_mp_empiric_parallel(self):
        """ MP Empiric not parallelized for dense matrices so far, fallback 
            to serial version. Until then, test is meaningless."""
        return self.skipTest("MP Empiric parallel: fallback to serial.")
        #=======================================================================
        # dist_s = mpe_s(self.dist)
        # dist_p = mutual_proximity_empiric(self.dist)
        # parallel_all_close_serial = np.allclose(dist_p, dist_s)
        # return self.assertTrue(parallel_all_close_serial)
        #=======================================================================
  
    def test_mp_empiric_sparse_parallel(self):
        """Test parallel version equivalent to serial version."""
        return self.skipTest("Function to test is deprecated.")
        #=======================================================================
        # sim = csr_matrix(1. - self.dist)
        # sim_s = mpe_s(sim, 'similarity')
        # sim_p = mutual_proximity_empiric(sim, 'similarity', verbose=1)
        # parallel_all_close_serial = np.allclose(sim_p.toarray(),
        #                                         sim_s.toarray())
        # return self.assertTrue(parallel_all_close_serial)
        #=======================================================================
  
    def test_mp_gauss_parallel(self):
        """Test parallel version equivalent to serial version."""
        return self.skipTest("MP Gauss not parallelized so far.")
   
    def test_mp_gauss_sparse_parallel(self):
        """Test parallel version equivalent to serial version."""
        return self.skipTest("MP Gauss not parallelized so far.")
   
    def test_mp_gaussi_parallel(self):
        """ MP GaussI not parallelized for dense matrices so far, fallback 
            to serial version. Until then, test is meaningless."""
        return self.skipTest("MP GaussI parallel: fallback to serial.")
        #=======================================================================
        # dist_s = mpmvni_s(self.dist)
        # dist_p = mutual_proximity_gaussi(self.dist)
        # parallel_all_close_serial = np.allclose(dist_p, dist_s)
        # return self.assertTrue(parallel_all_close_serial)
        #=======================================================================
  
    def test_mp_gaussi_sparse_parallel(self):
        """Test parallel version equivalent to serial version."""
        sim = csr_matrix(1. - self.dist)
        sim_s = mpmvni_s(sim, 'similarity')
        sim_p = mutual_proximity_gaussi(sim, 'similarity', mv=0, verbose=1)
        parallel_all_close_serial = np.allclose(sim_p.toarray(), 
                                                sim_s.toarray())
        return self.assertTrue(parallel_all_close_serial)
  
    def test_mp_gammai_parallel(self):
        """ MP GammaI not parallelized for dense matrices so far, fallback 
            to serial version. Until then, test is meaningless."""
        return self.skipTest("MP GammaI parallel: fallback to serial.")
        #=======================================================================
        # dist_s = mpgam_s(self.dist)
        # dist_p = mutual_proximity_gammai(self.dist)
        # parallel_all_close_serial = np.allclose(dist_p, dist_s)
        # return self.assertTrue(parallel_all_close_serial)
        #=======================================================================
  
    def test_mp_gammai_sparse_parallel(self):
        """Test parallel version equivalent to serial version."""
        sim = csr_matrix(1. - self.dist)
        sim_s = mpgam_s(sim, 'similarity')
        sim_p = mutual_proximity_gammai(sim, 'similarity', mv=0, verbose=1)
        parallel_all_close_serial = np.allclose(sim_p.toarray(),
                                                sim_s.toarray())
        return self.assertTrue(parallel_all_close_serial)

if __name__ == "__main__":
    unittest.main()
