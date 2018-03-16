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
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from hub_toolbox import approximate
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score

class Test(unittest.TestCase):


    def setUp(self):
        X, y = make_classification(
            n_samples=1_000, n_features=1_000, n_informative=500,
            n_redundant=0, n_repeated=0, n_classes=2,
            n_clusters_per_class=10, random_state=2847356)
        X_train, X_test, y_train, y_test = train_test_split(
            X.astype(np.float32), y.astype(np.int32), test_size=20)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.hr_algorithm = 'LS'
        self.n_neighbors = 5
        self.n_samples = 100
        self.sampling_algorithm = 'LSH'
        self.metric = 'sqeuclidean'
        self.n_jobs = 4
        self.verbose = 1


    def tearDown(self):
        pass


    def testApproximateHRwithLSH(self):
        hr = approximate.SuQHR(
            hr_algorithm=self.hr_algorithm, n_neighbors=self.n_neighbors,
            n_samples=self.n_samples, metric=self.metric,
            sampling_algorithm=self.sampling_algorithm,
            random_state=123, n_jobs=self.n_jobs, verbose=self.verbose)
        hr.fit(self.X_train, self.y_train)
        D_test_hr = hr.transform(self.X_test)
        D_test_hr = approximate.enforce_finite_distances(D_test_hr)
        y_pred = np.empty(self.X_test.shape[0])
        if hr.fixed_vantage_pts_:
            knn = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                       algorithm='brute',
                                       metric='precomputed')
            knn.fit(D_test_hr.T, hr.y_train_)
            y_pred = knn.predict(D_test_hr)
        else: # W/o fixed vantage points, a new classifier is required for each test object
            for i, d_test in enumerate(D_test_hr):
                d_test = d_test.reshape(-1, 1)
                knn = KNeighborsClassifier(n_neighbors=self.n_neighbors,
                                           algorithm='brute',
                                           metric='precomputed')
                knn.fit(d_test, self.y_train[hr.ind_test_[i]])
                y_pred[i] = knn.predict(d_test.T)
        acc = accuracy_score(y_pred, self.y_test)
        print(f'SuQHR accuracy: {acc:.2f}')

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
