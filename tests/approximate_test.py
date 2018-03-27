#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
import unittest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from hub_toolbox import approximate
from sklearn.metrics.classification import accuracy_score

class ApproximateHRTest(unittest.TestCase):


    def setUp(self):
        n_samples = 500
        n_informative = 256
        n_features = n_informative
        test_size = int(n_samples * .2)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=0, n_repeated=0, n_classes=2,
            n_clusters_per_class=10, random_state=2847356)
        X_train, X_test, y_train, y_test = train_test_split(
            X.astype(np.float32), y.astype(np.int32), test_size=test_size)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.hr_algorithms = ['LS', 'NICDM', 'MP', 'MPG', 'DSL', None]
        self.n_neighbors = 5
        self.n_samples = 100
        self.sampling_algorithms = ['random', 'kmeans++', 'LSH', 'HNSW', None]
        self.metrics = ['sqeuclidean', 'cosine']
        self.n_jobs = [-1, 1]
        self.verbose = 3
        self.accu_time = 0.

    def tearDown(self):
        print(f'Accumulated time: {self.accu_time} seconds.')

    def _approximate_hr(self, hr_algorithm, sampling_algorithm,
                        metric, n_jobs):
        hr = approximate.SuQHR(hr_algorithm=hr_algorithm,
                               n_neighbors=self.n_neighbors,
                               n_samples=self.n_samples,
                               metric=metric,
                               sampling_algorithm=sampling_algorithm,
                               random_state=123,
                               n_jobs=n_jobs,
                               verbose=self.verbose)
        hr.fit(self.X_train, self.y_train)
        y_pred = hr.predict(self.X_test)
        acc = accuracy_score(y_pred, self.y_test)
        print(f'SuQHR ({hr_algorithm}, {sampling_algorithm}, {metric}) '
              f'{self.n_neighbors}-NN accuracy: {acc:.2f}')
        total_time = hr.time_fit_ + hr.time_transform_ + hr.time_predict_
        self.accu_time += total_time.total.values

    def test_approximate_hubness_reduction(self):
        for hr_algorithm in self.hr_algorithms:
            for sampling_algorithm in self.sampling_algorithms:
                for metric in self.metrics:
                    for n_jobs in self.n_jobs:
                        self._approximate_hr(hr_algorithm,
                                             sampling_algorithm,
                                             metric,
                                             n_jobs)

    def test_surrogate_class(self):
        hr = approximate.ApproximateHubnessReduction()
        return self.assertIn(hr.hr_algorithm, self.hr_algorithms)

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
