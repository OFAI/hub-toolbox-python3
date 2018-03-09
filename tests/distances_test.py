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
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from hub_toolbox.distances import (cosine_distance, euclidean_distance,
                                   mp_dissim)
from hub_toolbox.io import load_dexter
from hub_toolbox.hubness import hubness

class TestDistances(unittest.TestCase):

    def setUp(self):
        np.random.seed(626)
        self.vectors = 99. * (np.random.rand(400, 200) - 0.5)

    def tearDown(self):
        del self.vectors

    def test_cosine_dist_equal_to_scipy_pdist_cos(self):
        cos_dist = cosine_distance(self.vectors)
        cos_dist_scipy = squareform(pdist(self.vectors, 'cosine'))
        result = np.allclose(cos_dist, cos_dist_scipy)
        return self.assertTrue(result)
    
    def test_euclidean_dist_equal_to_scipy_cdist_eucl(self):
        eucl_dist = euclidean_distance(self.vectors)
        eucl_dist_cdist = cdist(self.vectors, self.vectors, 'euclidean')
        result = np.allclose(eucl_dist, eucl_dist_cdist)
        return self.assertTrue(result)

class TestMpDisSim(unittest.TestCase):
    
    def setUp(self):
        _, y, X = load_dexter()
        r = np.random.permutation(y.size)
        self.X = X[r, :]
        self.y = y[r]
        split = int(len(y)/10*9)
        train_ind = slice(0, split)
        test_ind = slice(split, len(y))
        self.X_train = self.X[train_ind]
        self.X_test = self.X[test_ind]
        self.y_train = self.y[train_ind]
        self.y_test = self.y[test_ind]

    def test_mp_dissim(self):
        ''' Test that mp_dissim improves kNN-accuracy for dexter. '''
        D_part = cdist(self.X_test, self.X_train, 'euclidean')
        knn = KNeighborsClassifier(
            n_neighbors=5, metric='precomputed', n_jobs=4)
        knn.fit(self.X_train, self.y_train)
        y_pred = knn.predict(D_part)
        acc_eucl = accuracy_score(self.y_test, y_pred)
        h_eucl = hubness(D_part, k=5, metric='distance', n_jobs=4)[0]
        D_part_mp = mp_dissim(
            X=self.X_test, Y=self.X_train, p=0, n_bins=10, bin_size='r', verbose=1, n_jobs=-1)
        y_pred_mp = knn.predict(D_part_mp)
        acc_mp = accuracy_score(self.y_test, y_pred_mp)
        h_mp = hubness(D_part_mp, k=5, metric='distance', n_jobs=4)[0]
        #=======================================================================
        # print("Hub:", h_eucl, h_mp)
        # print("Acc:", acc_eucl, acc_mp)
        # D_mp = mp_dissim(self.X, p=2, n_bins=10, bin_size='r', n_jobs=-1, verbose=1)
        #=======================================================================
        self.assertLess(h_mp, h_eucl)
        self.assertGreater(acc_mp, acc_eucl)

if __name__ == "__main__":
    unittest.main()
