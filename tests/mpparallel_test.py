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
from hub_toolbox.Distances import euclidean_distance
from hub_toolbox.MutualProximity import mutual_proximity_empiric as mpe_s
from hub_toolbox.MutualProximity import mutual_proximity_gauss as mpmvn_s
from hub_toolbox.MutualProximity import mutual_proximity_gaussi as mpmvni_s
from hub_toolbox.MutualProximity import mutual_proximity_gammai as mpgam_s 
from hub_toolbox.MutualProximity_parallel import mutual_proximity_empiric, \
    mutual_proximity_gaussi, mutual_proximity_gammai
from scipy.sparse.csr import csr_matrix

class TestMutualProximityParallel(unittest.TestCase):
    """Unit tests for MutualProximity_parallel class"""

    def setUp(self):
        points = 5
        dim = 1000
        self.vector = 99. * (np.random.rand(points, dim) - 0.5)
        self.label = np.random.randint(0, 5, points)
        self.dist = euclidean_distance(self.vector)
        self.dist /= self.dist.max()

    def tearDown(self):
        del self.dist, self.label, self.vector

 #==============================================================================
 #    def test_mp_empiric_parallel(self):
 #        """ MP Empiric not parallelized for dense matrices so far, fallback 
 #            to serial version. Until then, test is meaningless."""
 #        dist_s = mpe_s(self.dist)
 #        dist_p = mutual_proximity_empiric(self.dist)
 #        parallel_all_close_serial = np.allclose(dist_p, dist_s)
 #        return self.assertTrue(parallel_all_close_serial)
 # 
 #    def test_mp_empiric_sparse_parallel(self):
 #        sim = csr_matrix(1. - self.dist)
 #        sim_s = mpe_s(sim, 'similarity')
 #        sim_p = mutual_proximity_empiric(sim, 'similarity')
 #        parallel_all_close_serial = np.allclose(sim_p.toarray(), 
 #                                                sim_s.toarray()) 
 #        return self.assertTrue(parallel_all_close_serial)
 # 
 #    def test_mp_gauss_parallel(self):
 #        """MP Gauss not parallelized so far, so skip this test for now."""
 #        pass
 # 
 #    def test_mp_gauss_sparse_parallel(self):
 #        """MP Gauss not parallelized so far, so skip this test for now."""
 #        pass
 # 
 #    def test_mp_gaussi_parallel(self):
 #        """ MP GaussI not parallelized for dense matrices so far, fallback 
 #            to serial version. Until then, test is meaningless."""
 #        dist_s = mpmvni_s(self.dist)
 #        dist_p = mutual_proximity_gaussi(self.dist)
 #        parallel_all_close_serial = np.allclose(dist_p, dist_s)
 #        return self.assertTrue(parallel_all_close_serial)
 #==============================================================================
 
    #===========================================================================
    # def test_mp_gaussi_sparse_parallel(self):
    #     sim = csr_matrix(1. - self.dist)
    #     sim_s = mpmvni_s(sim, 'similarity')
    #     sim_p = mutual_proximity_gaussi(sim, 'similarity')
    #     parallel_all_close_serial = np.allclose(sim_p.toarray(), 
    #                                             sim_s.toarray())
    #     return self.assertTrue(parallel_all_close_serial)
    #===========================================================================
 
    def test_mp_gammai_parallel(self):
        """ MP GammaI not parallelized for dense matrices so far, fallback 
            to serial version. Until then, test is meaningless."""
        dist_s = mpgam_s(self.dist)
        dist_p = mutual_proximity_gammai(self.dist)
        parallel_all_close_serial = np.allclose(dist_p, dist_s)
        return self.assertTrue(parallel_all_close_serial)
 
    def test_mp_gammai_sparse_parallel(self):
        sim = csr_matrix(1. - self.dist)
        sim_s = mpgam_s(sim, 'similarity')
        sim_p = mutual_proximity_gammai(sim, 'similarity')
        d = (sim_s - sim_p).toarray()
        parallel_all_close_serial = np.allclose(sim_p.toarray(),
                                                sim_s.toarray())
        return self.assertTrue(parallel_all_close_serial)

if __name__ == "__main__":
    unittest.main()
