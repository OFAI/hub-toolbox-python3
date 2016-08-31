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
from scipy.spatial.distance import squareform, pdist
from hub_toolbox.GoodmanKruskal import goodman_kruskal_index,\
    _naive_goodman_kruskal, sparse_goodman_kruskal_index
from hub_toolbox.IO import random_sparse_matrix
from hub_toolbox.SharedNN import shared_nearest_neighbors
from scipy.sparse.csr import csr_matrix

class TestGoodmanKruskal(unittest.TestCase):

    def setUp(self):
        n = 50
        m = 5
        c = 3
        data = np.random.rand(n, m)
        self.distance = squareform(pdist(data, 'euclidean'))
        self.similarity = 1. - self.distance / self.distance.max()
        self.labels = np.random.randint(0, c, n)

    def tearDown(self):
        del self.distance, self.similarity, self.labels
        
    def test_naive_goodmankruskal_algorithm(self):
        """Using a small clustering with correct value calc by hand"""
        distance = np.array(
            squareform([0.7, 1.55, 0.5, 1.7, 0.9, 0.85, 1.2, 1.5, 0.6, 1.4]))
        label = np.array([0, 0, 1, 2, 1])
        CORRECT_RESULT = 0.75
        result = _naive_goodman_kruskal(distance, label, 'distance')
        return self.assertEqual(result, CORRECT_RESULT)

    def test_efficient_goodmankruskal_equal_to_naive_goodmankruskal(self):
        """Test whether goodman_kruskal_index yields correct result"""
        gamma_efficient = goodman_kruskal_index(self.distance, self.labels)
        gamma_naive = _naive_goodman_kruskal(self.distance, self.labels)
        return self.assertEqual(gamma_efficient, gamma_naive)
    
    def test_goodmankruskal_distance_based_equal_to_similarity_based(self):
        """Test whether results are correct using similarities"""
        gamma_dist = goodman_kruskal_index(self.distance, self.labels, 'distance')
        gamma_sim = goodman_kruskal_index(self.similarity, self.labels, 'similarity')
        return self.assertEqual(gamma_dist, gamma_sim)
    
    def test_goodmankruskal_close_to_zero_for_random_data(self):
        gamma_dist = goodman_kruskal_index(self.distance, self.labels)
        return self.assertAlmostEqual(gamma_dist, 0., places=1)
    
    def test_sparse_goodmankruskal_equal_to_dense_goodmankruskal(self):
        similarity = random_sparse_matrix(size=1000)
        labels = np.random.randint(0, 5, 1000)
        gamma_sparse = sparse_goodman_kruskal_index(similarity, labels, verbose=1)
        gamma_dense = goodman_kruskal_index(similarity.toarray(), labels, 'similarity')
        return self.assertEqual(gamma_dense, gamma_sparse)
    
    def test_correct_handling_equal_distances_goodmankruskal(self):
        """SharedNN matrices contain lots of equal distances"""
        dist_snn = shared_nearest_neighbors(self.distance)
        gamma_efficient = goodman_kruskal_index(dist_snn, self.labels)
        gamma_naive = _naive_goodman_kruskal(dist_snn, self.labels)
        return self.assertEqual(gamma_efficient, gamma_naive)
    
    def test_correct_handling_equal_similarities_sparse_gk(self):
        sim_snn = 1. - shared_nearest_neighbors(self.distance)
        gamma_sparse = sparse_goodman_kruskal_index(csr_matrix(sim_snn), self.labels)
        gamma_efficient = goodman_kruskal_index(sim_snn, self.labels, 'similarity')
        return self.assertEqual(gamma_efficient, gamma_sparse)

if __name__ == "__main__":
    unittest.main()
