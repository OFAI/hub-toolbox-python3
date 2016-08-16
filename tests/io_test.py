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
from scipy.sparse.csr import csr_matrix
from hub_toolbox.IO import random_sparse_matrix

class TestIO(unittest.TestCase):

    def setUp(self):
        np.random.seed(626)
        self.matrix_n = 500
        self.density = 0.02
        self.similarity = random_sparse_matrix(
            size=self.matrix_n, density=self.density)

    def tearDown(self):
        del self.matrix_n, self.density, self.similarity

    def test_random_sparse_similarity_matrix_quadratic_form(self):
        return self.assertEqual(
            self.similarity.shape[0], self.similarity.shape[1])

    def test_random_sparse_similarity_matrix_correct_size(self):
        return self.assertEqual(self.similarity.shape[0], self.matrix_n)

    def test_random_sparse_similarity_matrix_correct_type(self):
        return self.assertIsInstance(self.similarity, csr_matrix)

    def test_random_sparse_similarity_matrix_symmetric(self):
        non_symmetric_entry = \
            (self.similarity - self.similarity.T != 0.).nnz > 0
        return self.assertFalse(non_symmetric_entry)

    def test_random_sparse_similarity_matrix_min_zero(self):
        return self.assertGreaterEqual(self.similarity.min(), 0.)

    def test_random_sparse_similarity_matrix_max_one(self):
        return self.assertLessEqual(self.similarity.max(), 1.)

    def test_random_sparse_similarity_matrix_self_similarity_one(self):
        all_diag_ones = np.all(self.similarity.diagonal() == 1)
        return self.assertTrue(all_diag_ones)

    def test_random_sparse_similarity_matrix_density(self):
        return self.assertAlmostEqual(
            self.similarity.nnz / self.matrix_n**2, self.density*2, places=2)

if __name__ == "__main__":
    unittest.main()
