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
try: # for scikit-learn >= 0.18
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
except ImportError: # lower scikit-learn versions
    from sklearn.cross_validation import LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score as f1_score_sklearn
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from hub_toolbox.Distances import sample_distance
from hub_toolbox.IO import load_dexter
from hub_toolbox.KnnClassification import score, predict, f1_score
from hub_toolbox.KnnClassification import f1_macro, f1_micro, f1_weighted

class TestKnnClassification(unittest.TestCase):

    def setUp(self):
        self.distance, self.label, self.vector = load_dexter()
        self.n = self.distance.shape[0]

    def tearDown(self):
        del self.distance, self.label, self.vector

    def test_knn_predict_equal_sklearn_loocv_predict(self):
        y = LabelEncoder().fit_transform(self.label)
        y_pred = predict(self.distance, y, k=5, 
                         metric='distance', return_cmat=False)[0].ravel()
        knn = KNeighborsClassifier(
            n_neighbors=5, algorithm='brute', metric='precomputed')
        n = self.distance.shape[0] # for LOO-CV
        try: # sklearn < 0.18
            loo_cv = LeaveOneOut(n)
        except TypeError:
            loo_cv = LeaveOneOut()
        y_pred_sklearn = cross_val_predict(
            knn, self.distance, y, cv=loo_cv)
        return self.assertTrue(np.alltrue(y_pred == y_pred_sklearn))
        
    def test_f1_score(self):
        y = LabelBinarizer().fit_transform(self.label).ravel()
        y_pred, cmat = predict(self.distance, y, k=5, metric='distance')
        y_pred = y_pred.ravel()
        knn = KNeighborsClassifier(
            n_neighbors=5, algorithm='brute', metric='precomputed')
        n = self.distance.shape[0] # for LOO-CV
        try: # sklearn < 0.18
            loo_cv = LeaveOneOut(n)
        except TypeError:
            loo_cv = LeaveOneOut()
        y_pred_sklearn = cross_val_predict(
            knn, self.distance, y, cv=loo_cv)
        f1_binary_hub = f1_score(cmat[0, 0, :, :])
        f1_binary_sklearn = f1_score_sklearn(y, y_pred_sklearn, average='binary')
        return self.assertEqual(f1_binary_hub, f1_binary_sklearn)

    def test_f1_micro_macro_weighted(self):
        y = np.random.randint(0, 5, self.label.size).reshape(-1, 1)
        y = OneHotEncoder().fit_transform(y).toarray()
        y_pred, cmat = predict(self.distance, y, k=5, metric='distance')
        y_pred = y_pred[0]
        knn = KNeighborsClassifier(
            n_neighbors=5, algorithm='brute', metric='precomputed')
        n = self.distance.shape[0] # for LOO-CV
        try: # sklearn < 0.18
            loo_cv = LeaveOneOut(n)
        except TypeError:
            loo_cv = LeaveOneOut()
        y_pred_sklearn = cross_val_predict(
            knn, self.distance, y, cv=loo_cv)
        f1_hub = [f1_macro(cmat[0]), f1_micro(cmat[0]), f1_weighted(cmat[0])]
        f1_sklearn = [f1_score_sklearn(y, y_pred_sklearn, average='macro'),
                      f1_score_sklearn(y, y_pred_sklearn, average='micro'),
                      f1_score_sklearn(y, y_pred_sklearn, average='weighted')]
        return self.assertListEqual(f1_hub, f1_sklearn)

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
        try: # sklearn < 0.18
            loo_cv = LeaveOneOut(n)
        except TypeError:
            loo_cv = LeaveOneOut()
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

    def test_sample_knn(self):
        """ Make sure that sample-kNN works correctly. """
        # TODO create a stricter test
        X = np.array([[1., 2.],
                      [2., 2.],
                      [2., 3.],
                      [3., .5],
                      [4., 1.5]])
        y = np.array([0, 1, 0, 1, 1])
        s = 2
        rnd = 1234
        D, sample_idx = sample_distance(X, y, s, random_state=rnd)
        expected_sample_idx = np.array([4, 2])
        expected_acc = 0.4
        if not np.setdiff1d(sample_idx, expected_sample_idx).size == \
               np.setdiff1d(expected_sample_idx, sample_idx).size == 0:
            return self.fail("Test implementation broken: wrong sample.")
        acc, _, _ = score(D=D, target=y, k=2, metric='distance', 
                          sample_idx=sample_idx)
        return self.assertEqual(expected_acc, acc[0, 0])

if __name__ == "__main__":
    unittest.main()
