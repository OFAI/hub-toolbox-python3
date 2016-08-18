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
from hub_toolbox.MutualProximity import mutual_proximity_empiric,\
    mutual_proximity_gauss, mutual_proximity_gaussi, mutual_proximity_gammai
from scipy.sparse.csr import csr_matrix

class TestMutualProximity(unittest.TestCase):
    """Unit tests for the MutualProximity class (serial computing)"""

    def setUp(self):
        np.random.seed(626)
        points = 5
        dim = 100
        self.vector = 99. * (np.random.rand(points, dim) - 0.5)
        self.label = np.random.randint(0, 5, points)
        self.dist = euclidean_distance(self.vector)
        # scale to [0, 1), avoiding 1: otherwise sparseMP != denseMP (by design)
        self.dist /= (self.dist.max() + 1e-12)

    def tearDown(self):
        del self.dist, self.label, self.vector

    #===========================================================================
    # def test_mp_empiric(self):
    #     """Test MP Empiric for toy example (ground truth calc by hand)"""
    #     return self.fail()
    #===========================================================================
  
    def test_mp_empiric_dist_equal_sim(self):
        sim = 1. - self.dist
        mp_dist = mutual_proximity_empiric(self.dist, 'distance')
        mp_sim = mutual_proximity_empiric(sim, 'similarity')
        dist_allclose_one_minus_sim = np.allclose(mp_dist, 1. - mp_sim)
        return self.assertTrue(dist_allclose_one_minus_sim)
  
    def test_mp_empiric_sparse_equal_dense(self):
        sim_dense = 1. - self.dist
        sim_sparse = csr_matrix(sim_dense)
        mp_dense = mutual_proximity_empiric(sim_dense, 'similarity')
        mp_sparse = mutual_proximity_empiric(sim_sparse, 'similarity')
        dense_allclose_sparse = np.allclose(mp_dense, mp_sparse.toarray())
        return self.assertTrue(dense_allclose_sparse)
  
    #===========================================================================
    # def test_mp_gauss(self):
    #     """Test MP Gauss for toy example (ground truth calc by hand)"""
    #     return self.fail()
    #===========================================================================
  
    def test_mp_gauss_dist_equal_sim(self):
        sim = 1. - self.dist
        mp_dist = mutual_proximity_gauss(self.dist, 'distance')
        mp_sim = mutual_proximity_gauss(sim, 'similarity')
        dist_allclose_one_minus_sim = np.allclose(mp_dist, 1. - mp_sim)
        return self.assertTrue(dist_allclose_one_minus_sim)
  
    def test_mp_gauss_sparse_equal_dense(self):
        """MP Gauss not implemented for sparse matrices so far."""
        pass
  
    
    #===========================================================================
    # def test_mp_gaussi(self):
    #     """Test MP GaussI for toy example (ground truth calc by hand)"""
    #     return self.fail()
    #===========================================================================
  
    def test_mp_gaussi_dist_equal_sim(self):
        sim = 1. - self.dist
        mp_dist = mutual_proximity_gaussi(self.dist, 'distance')
        mp_sim = mutual_proximity_gaussi(sim, 'similarity')
        dist_allclose_one_minus_sim = np.allclose(mp_dist, 1. - mp_sim)
        return self.assertTrue(dist_allclose_one_minus_sim)
 
    def test_mp_gaussi_sparse_equal_dense(self):
        sim_dense = 1. - self.dist
        sim_sparse = csr_matrix(sim_dense)
        mp_dense = mutual_proximity_gaussi(sim_dense, 'similarity')
        mp_sparse = mutual_proximity_gaussi(sim_sparse, 'similarity')
        dense_allclose_sparse = np.allclose(mp_dense, mp_sparse.toarray())
        return self.assertTrue(dense_allclose_sparse)

 #==============================================================================
 #    def test_mp_gammai(self):
 #        """Test MP GammaI for toy example (ground truth calc by hand)"""
 #        return self.fail()
 # 
 #==============================================================================
 
 #==============================================================================
 # 
 #    def test_mp_gammai_dist_equal_sim(self):
 #        sim = 1. - self.dist
 #        mp_dist = mutual_proximity_gammai(self.dist, 'distance')
 #        mp_sim = mutual_proximity_gammai(sim, 'similarity')
 #        dist_allclose_one_minus_sim = np.allclose(mp_dist, 1. - mp_sim)
 #        return self.assertTrue(dist_allclose_one_minus_sim)
 #==============================================================================
 
    def test_mp_gammai_sparse_equal_dense(self):
        sim_dense = 1. - self.dist
        sim_sparse = csr_matrix(sim_dense)
        mp_dense = mutual_proximity_gammai(sim_dense, 'similarity')
        mp_sparse = mutual_proximity_gammai(sim_sparse, 'similarity')
        dense_allclose_sparse = np.allclose(mp_dense, mp_sparse.toarray())
        return self.assertTrue(dense_allclose_sparse)
    
if __name__ == "__main__":
    unittest.main()
