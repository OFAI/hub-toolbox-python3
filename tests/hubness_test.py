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
from scipy import sparse
from scipy.spatial.distance import squareform
from sklearn.datasets.samples_generator import make_classification
from sklearn.model_selection import train_test_split
from hub_toolbox.approximate import ApproximateHubnessReduction,\
                                    VALID_HR, VALID_SAMPLE
from hub_toolbox.distances import euclidean_distance
from hub_toolbox.hubness import hubness, Hubness, hubness_from_vectors
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
        hub.fit_transform(self.D, has_self_distances=True)
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

    def test_hubness_from_sparse_precomputed_matrix(self):
        # Generate high-dimensional data
        X, y = make_classification(n_samples=1000,
                                   n_features=100,
                                   n_informative=100,
                                   n_redundant=0,
                                   n_repeated=0,
                                   random_state=123)
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        for hr_algorithm in VALID_HR: #['dsl']:#
            for sampling_algorithm in VALID_SAMPLE: #['hnsw', 'lsh']:#
                for n_samples in [50, 100]:
                    print(f'Test {hr_algorithm}, {sampling_algorithm}, '
                          f'with {n_samples} samples.')
                    self.hubness_from_sparse_precomputed_matrix(
                        X, y, hr_algorithm, sampling_algorithm, n_samples)
        
    def hubness_from_sparse_precomputed_matrix(
            self, X, y, hr, sample, n_samples):
        # Make train-test split
        X_train, X_test, y_train, _ = train_test_split(X, y)
        #print(f"n_train={X_train.shape[0]}, n_test={X_test.shape[0]}, "
        #      f"HR={hr}, sampling={sample}, n_samples={n_samples}.")
        # Obtain a sparse distance matrix
        ahr = ApproximateHubnessReduction(
            hr_algorithm=hr, sampling_algorithm=sample, n_samples=n_samples)
        ahr.fit(X_train, y_train)
        _ = ahr.transform(X_test)
        D_test_csr = ahr.sec_dist_sparse_
        # Hubness in sparse matrix
        hub = Hubness(k=10,
                      metric='precomputed',
                      return_k_neighbors=True,
                      shuffle_equal=False)
        hub.fit_transform(D_test_csr)
        Sk_trunc_sparse = hub.k_skewness_truncnorm_
        Sk_sparse = hub.k_skewness_
        k_neigh_sparse = hub.k_neighbors_
        # Hubness in dense matrix
        try:
            D_test_dense = D_test_csr.toarray()
        except AttributeError:
            return # Without sampling, the distance matrix is not sparse
        D_test_dense[D_test_dense == 0] = np.finfo(np.float32).max
        hub_dense = Hubness(k=10,
                            metric='precomputed',
                            return_k_neighbors=True,
                            shuffle_equal=False)
        hub_dense.fit_transform(D_test_dense)
        Sk_trunc_dense = hub_dense.k_skewness_truncnorm_
        Sk_dense = hub_dense.k_skewness_
        k_neigh_dense = hub_dense.k_neighbors_
        if hr in ['MP', 'MPG']:
            decimal = 1
        else:
            decimal = 5
        try:
            np.testing.assert_array_equal(
                k_neigh_dense.ravel(), k_neigh_sparse)
        except AssertionError:
            s1 = k_neigh_dense.sum()
            s2 = k_neigh_sparse.sum()
            sm = max(s1, s2)
            print(f'k_neighbors not identical, but close: '
                  f'{s1}, {s2}, {s1/s2}.')
            np.testing.assert_allclose(s2/sm, s1/sm, rtol=1e-2)
        np.testing.assert_array_almost_equal(
            Sk_sparse, Sk_dense, decimal=decimal)
        np.testing.assert_array_almost_equal(
            Sk_trunc_sparse, Sk_trunc_dense, decimal=decimal)

if __name__ == "__main__":
    unittest.main()
