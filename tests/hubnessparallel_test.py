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
from hub_toolbox.HubnessAnalysis import load_dexter
from hub_toolbox.Hubness_parallel import hubness as hubness_p
from hub_toolbox.Hubness import hubness as hubness_s
from hub_toolbox.IO import random_sparse_matrix

class TestHubnessParallel(unittest.TestCase):
    """Test that parallelized hubness yields same results as serial version"""

    def setUp(self):
        self.distance, self.label, self.vector = load_dexter()

    def tearDown(self):
        del self.distance, self.label, self.vector

    def test_parallel_hubness_equal_serial_hubness_distance_based(self):
        S_k_p, D_k_p, N_k_p = hubness_p(
            self.distance, k=5, metric='distance', verbose=True, n_jobs=2)
        S_k_s, D_k_s, N_k_s = hubness_s(
            self.distance, k=5, metric='distance', verbose=False)
        result = np.allclose(S_k_p, S_k_s) & \
            np.allclose(D_k_p, D_k_s) & \
            np.allclose(N_k_p, N_k_s)
        return self.assertTrue(result)

    def test_parallel_hubness_equal_serial_hubness_similarity_based(self):
        similarity = random_sparse_matrix(size=1000)
        S_k_p, D_k_p, N_k_p = hubness_p(
            similarity, k=5, metric='similarity', verbose=False, n_jobs=-1)
        S_k_s, D_k_s, N_k_s = hubness_s(
            similarity, k=5, metric='similarity', verbose=False)
        result = np.allclose(S_k_p, S_k_s) & \
            np.allclose(D_k_p, D_k_s) & \
            np.allclose(N_k_p, N_k_s)
        return self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
