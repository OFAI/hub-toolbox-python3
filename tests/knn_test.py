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
from sklearn.cross_validation import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from hub_toolbox.HubnessAnalysis import load_dexter
from hub_toolbox.KnnClassification import score

class TestKnnClassification(unittest.TestCase):

    def setUp(self):
        self.distance, self.label, self.vector = load_dexter()
        self.n = self.distance.shape[0]

    def tearDown(self):
        del self.distance, self.label, self.vector

    def test_knn_score_matches_correct_prediction_fraction(self):
        k = np.array([1, 5, 20])
        acc, correct, _ = score(self.distance, self.label, k=k)
        acc_match = np.zeros_like(k, dtype=bool)
        for i, _ in enumerate(k):
            cur_acc = acc[i]
            cur_correct = correct[i]
            acc_match[i] = np.allclose(cur_acc, cur_correct.sum() / self.n)
        return self.assertTrue(np.all(acc_match))

    def test_knn_score_matches_confusion_matrix(self):
        k = np.array([1, 5, 20])
        acc, _, cmat = score(self.distance, self.label, k=k)
        acc_match = np.zeros_like(k, dtype=bool)
        for i, _ in enumerate(k):
            cur_acc = acc[i]
            cur_cmat = cmat[i]
            TP = cur_cmat[0, 0]
            FN = cur_cmat[0, 1]
            FP = cur_cmat[1, 0]
            TN = cur_cmat[1, 1]
            acc_from_cmat = (TP + TN) / (TP + FN + FP + TN)
            acc_match[i] = np.allclose(cur_acc, acc_from_cmat)
        return self.assertTrue(np.all(acc_match))

    def test_knn_score_equal_sklearn_loocv_score(self):
        acc, correct, cmat = \
            score(self.distance, self.label, k=5, metric='distance')
        # scoring only one k value, so take just the first elements:
        acc = acc[0, 0]
        correct = correct[0]
        cmat = cmat[0]
        # This should work too, but is much slower than using precomp. dist.
        #=======================================================================
        # knclassifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute', 
        #                                     metric='cosine')
        #=======================================================================
        knclassifier = KNeighborsClassifier(n_neighbors=5, algorithm='brute', 
                                            metric='precomputed')
        n = self.distance.shape[0] # for LOO-CV
        loo_cv = LeaveOneOut(n)
        predicted_sklearn = cross_val_predict(
            knclassifier, self.distance, self.label, cv=loo_cv)
        acc_sklearn = accuracy_score(self.label, predicted_sklearn)
        if not np.allclose(acc, acc_sklearn):
            return self.assertAlmostEqual(acc, acc_sklearn, places=7)
        else:
            correct_sklearn = predicted_sklearn == self.label
            equal_prediction = np.all(correct == correct_sklearn)
            msg = """Accuracies of hub toolbox k-NN and sklearn-kNN are almost 
                     equal, but the predictions per data point are not."""
            return self.assertTrue(equal_prediction, msg)

if __name__ == "__main__":
    unittest.main()
