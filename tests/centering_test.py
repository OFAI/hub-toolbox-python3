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
from sklearn.preprocessing import StandardScaler
from hub_toolbox.centering import centering, weighted_centering, \
    localized_centering, dis_sim_global, dis_sim_local
from hub_toolbox.io import load_dexter
from hub_toolbox.hubness import hubness
from hub_toolbox.knn_classification import score

class TestCentering(unittest.TestCase):
    
    def setUp(self):
        self.distance, self.target, self.vectors = load_dexter()

    def test_centering_equal_to_sklearn_centering(self):
        vectors_cent = centering(self.vectors, 'vector')
        scaler = StandardScaler(with_mean=True, with_std=False)
        vectors_sklearn_cent = scaler.fit_transform(self.vectors)
        result = np.allclose(vectors_cent, vectors_sklearn_cent, rtol=1e-7)
        return self.assertTrue(result)

    def test_weighted_centering_with_gamma_zero_equal_centering(self):
        vectors_wcent = weighted_centering(self.vectors, 'cosine', gamma=0.)
        vectors_cent = centering(self.vectors, 'vector')
        result = np.allclose(vectors_cent, vectors_wcent)
        return self.assertTrue(result)

    def test_weighted_centering_with_gamma_notzero_changes_result(self):
        gamma = np.random.rand(1)
        vectors_wcent = weighted_centering(self.vectors, 'cosine', gamma)
        vectors_cent = centering(self.vectors, 'vector')
        result = np.allclose(vectors_cent, vectors_wcent)
        return self.assertFalse(result)

    def test_localized_centering(self):
        """Test whether hubness and k-NN accuracy improve for dexter"""
        h_orig = hubness(self.distance)[0]
        acc_orig = score(self.distance, self.target)[0][0, 0]
        sim_lcent = localized_centering(self.vectors, kappa=20, gamma=1.)
        h_lcent = hubness(sim_lcent, metric='similarity')[0]
        acc_lcent = score(sim_lcent, self.target, metric='similarity')[0][0, 0]
        result = (h_orig / h_lcent > 1.5) & (acc_lcent - acc_orig > 0.03)
        return self.assertTrue(result)

    def test_localized_centering_parallel(self):
        lcent_seq = localized_centering(
            self.vectors, kappa=20, gamma=1., n_jobs=4)
        lcent_par = localized_centering(
            self.vectors, kappa=20, gamma=1., n_jobs=1)
        return np.testing.assert_array_almost_equal(lcent_par, lcent_seq, 14)

    def test_dis_sim_global(self):
        """Test whether hubness and k-NN accuracy improve for dexter"""
        h_orig = hubness(self.distance)[0]
        acc_orig = score(self.distance, self.target)[0][0, 0]
        dist_dsg = dis_sim_global(self.vectors)
        h_dsg = hubness(dist_dsg)[0]
        acc_dsg = score(dist_dsg, self.target)[0][0, 0]
        result = (h_orig / h_dsg > 2) & (acc_dsg - acc_orig > 0.07)
        return self.assertTrue(result)

    def test_dis_sim_local(self):
        """Test whether hubness and k-NN accuracy improve for dexter"""
        #self.vectors = np.tile(self.vectors, 1)
        h_orig = hubness(self.distance)[0]
        acc_orig = score(self.distance, self.target)[0][0, 0]
        dist_dsl = dis_sim_local(self.vectors, k=50)
        h_dsl = hubness(dist_dsl)[0]
        acc_dsl = score(dist_dsl, self.target)[0][0, 0]
        result = (h_orig / h_dsl > 10) & (acc_dsl - acc_orig > 0.03)
        return self.assertTrue(result)

    def test_dis_sim_local_parallel(self):
        dsl_seq = dis_sim_local(self.vectors, k=50, n_jobs=1)
        dsl_par = dis_sim_local(self.vectors, k=50, n_jobs=4)
        return np.testing.assert_array_almost_equal(dsl_seq, dsl_par, 14)

    def test_dis_sim_local_split_parallel_(self):
        X = self.vectors[:150, :]
        Y = self.vectors[150:, :]
        dsl_seq = dis_sim_local(X, Y, n_jobs=1)
        dsl_par = dis_sim_local(X, Y, n_jobs=4)
        return np.testing.assert_array_almost_equal(dsl_seq, dsl_par, 14)

if __name__ == "__main__":
    unittest.main()
