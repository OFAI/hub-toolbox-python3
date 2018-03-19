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
from scipy.spatial.distance import squareform
from hub_toolbox.hubness import hubness, Hubness, hubness_from_vectors
from hub_toolbox.distances import euclidean_distance
from hub_toolbox.io import random_sparse_matrix

class TestHubness(unittest.TestCase):
    """Test hubness calculations"""

    def setUp(self):
        """Hubness truth: S_k=5, skewness calculated with bias"""
        self.dist = squareform([.2, .1, .8, .4, .3, .5, .7, 1., .6, .9])
        self.hubness_truth = -0.2561204163

    def tearDown(self):
        del self.dist

    def test_hubness(self):
        """Test hubness against ground truth calc on spreadsheet"""
        Sk5, _, _ = hubness(self.dist, k=2, verbose=1)
        return self.assertAlmostEqual(Sk5, self.hubness_truth, places=10)

    def test_hubness_return_values_are_self_consistent(self):
        """Test that the three returned values fit together"""
        np.random.seed(626)
        points = 200
        dim = 500
        vector = 99. * (np.random.rand(points, dim) - 0.5)
        dist = euclidean_distance(vector)
        k = 10
        Sk10, Dk10, Nk10 = hubness(dist, k=k)
        # Dk is just checked for correct shape
        correct_dim_Dk10 = Dk10.shape == (points, k)
        # Count k-occurence (different method than in module)
        Dk10 = Dk10.ravel()
        Nk10_true = np.zeros(points, dtype=int)
        for i in range(points):
            Nk10_true[i] = (Dk10 == i).sum()
        correct_Nk10 = np.all(Nk10 == Nk10_true)
        # Calculate skewness (different method than in module)
        x0 = Nk10 - Nk10.mean()
        s2 = (x0**2).mean()
        m3 = (x0**3).mean()
        s = m3 / (s2**1.5)
        Sk10_true = s
        correct_Sk10 = Sk10 == Sk10_true
        return self.assertTrue(correct_dim_Dk10
                               and correct_Nk10
                               and correct_Sk10)

    def test_parallel_hubness_equal_serial_hubness_distance_based(self):
        S_k_p, D_k_p, N_k_p = hubness(
            self.dist, k=5, metric='distance', verbose=True, n_jobs=-1)
        S_k_s, D_k_s, N_k_s = hubness(
            self.dist, k=5, metric='distance', verbose=False, n_jobs=1)
        np.testing.assert_array_almost_equal(S_k_p, S_k_s, decimal=7)
        np.testing.assert_array_almost_equal(D_k_p, D_k_s, decimal=7)
        np.testing.assert_array_almost_equal(N_k_p, N_k_s, decimal=7)

    def test_parallel_hubness_equal_serial_hubness_similarity_based(self):
        similarity = random_sparse_matrix(size=1000)
        S_k_p, D_k_p, N_k_p = hubness(
            similarity, k=5, metric='similarity', verbose=False, n_jobs=-1)
        S_k_s, D_k_s, N_k_s = hubness(
            similarity, k=5, metric='similarity', verbose=False, n_jobs=1)
        np.testing.assert_array_almost_equal(S_k_p, S_k_s, decimal=7)
        np.testing.assert_array_almost_equal(D_k_p, D_k_s, decimal=7)
        np.testing.assert_array_almost_equal(N_k_p, N_k_s, decimal=7)

class TestHubnessClass(unittest.TestCase):
    """Test hubness calculations"""

    def setUp(self):
        """Hubness truth: S_k=5, skewness calculated with bias"""
        np.random.seed(123)
        self.X = np.random.rand(100, 50)
        self.D = euclidean_distance(self.X)

    def tearDown(self):
        del self.X

    def test_hubness_against_distance(self):
        """Test hubness class against distance-based methods."""
        Sk_dist, Dk_dist, Nk_dist = hubness(self.D, k=10)
        hub = Hubness(k=10, return_k_neighbors=True, return_k_occurrence=True)
        hub.fit_transform(self.X)
        Sk_class = hub.k_skewness_
        Dk_class = hub.k_neighbors_
        Nk_class = hub.k_occurrence_
        np.testing.assert_almost_equal(Sk_class, Sk_dist, decimal=10)
        np.testing.assert_array_equal(Dk_class, Dk_dist)
        np.testing.assert_array_equal(Nk_class, Nk_dist)
        hub = Hubness(k=10, return_k_neighbors=True, return_k_occurrence=True,
                      metric='precomputed')
        hub.fit_transform(self.D)
        Sk_class = hub.k_skewness_
        Dk_class = hub.k_neighbors_
        Nk_class = hub.k_occurrence_
        np.testing.assert_almost_equal(Sk_class, Sk_dist, decimal=10)
        np.testing.assert_array_equal(Dk_class, Dk_dist)
        np.testing.assert_array_equal(Nk_class, Nk_dist)

    def test_hubness_against_vectors(self):
        """ Test hubness class against vector-based method. """
        Sk_vect, Dk_vect, Nk_vect = hubness_from_vectors(self.X, k=10)
        hub = Hubness(k=10, return_k_neighbors=True, return_k_occurrence=True)
        hub.fit_transform(self.X)
        Sk_class = hub.k_skewness_
        Dk_class = hub.k_neighbors_
        Nk_class = hub.k_occurrence_
        np.testing.assert_almost_equal(Sk_class, Sk_vect, decimal=10)
        np.testing.assert_array_equal(Dk_class, Dk_vect)
        np.testing.assert_array_equal(Nk_class, Nk_vect)
        np.testing.assert_array_less(
            hub.k_skewness_truncnorm_, hub.k_skewness_)

    def test_hubness_independent_on_data_set_size(self):
        thousands = 3
        n_objects = thousands * 1_000
        X = np.random.rand(n_objects, 128)
        N_SAMPLES = np.arange(1, thousands + 1) * 1_000
        Sk_trunc = np.empty(N_SAMPLES.size)
        for i, n_samples in enumerate(N_SAMPLES):
            ind = np.random.permutation(n_objects)[:n_samples]
            X_sample = X[ind, :]
            hub = Hubness()
            hub.fit_transform(X_sample)
            Sk_trunc[i] = hub.k_skewness_truncnorm_
            if i > 0:
                np.testing.assert_allclose(
                    Sk_trunc[i], Sk_trunc[i-1], rtol=1e-1, 
                    err_msg=(f'Hubness measure is too dependent on data set '
                             f'size with S({N_SAMPLES[i]}) = x '
                             f'and S({N_SAMPLES[i-1]}) = y.'))
        np.testing.assert_allclose(Sk_trunc[-1], Sk_trunc[0], rtol=1e-1)


if __name__ == "__main__":
    unittest.main()
