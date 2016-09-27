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
from hub_toolbox.IO import load_dexter
from hub_toolbox.IntrinsicDim import intrinsic_dimension

class TestIntrinsicDim(unittest.TestCase):

    def setUp(self):
        self.vector = np.random.rand(50, 2)

    def tearDown(self):
        del self.vector

    def test_intrinsic_dim_mle_levina(self):
        """Test against value calc. by matlab reference implementation."""
        _, _, vector = load_dexter()
        ID_MLE_REF = 74.742
        id_mle = intrinsic_dimension(vector, k1=6, k2=12, 
            estimator='levina', metric='vector', trafo=None)
        return self.assertEqual(id_mle, int(ID_MLE_REF))

    def test_intrinsic_dim_mle_levina_low_memory(self):
        """ Same as above, but invoking the speed-memory trade-off. """
        _, _, vector = load_dexter()
        ID_MLE_REF = 74.742
        id_mle = intrinsic_dimension(vector, 6, 12, 'levina', 
                                     'vector', None, mem_threshold=0)
        return self.assertEqual(id_mle, int(ID_MLE_REF))

    def test_incorrect_est_params(self):
        """ Test handling of incorrect estimator. """
        with self.assertRaises(ValueError):
            intrinsic_dimension(self.vector, 
                estimator='the_single_truly_best_id_estimator')

    def test_incorrect_k1_params(self):
        """ Test handling of incorrect neighborhood parameters."""
        with self.assertRaises(ValueError):
            intrinsic_dimension(self.vector, k1=0)

    def test_incorrect_k12_params(self):
        """ Test handling of incorrect neighborhood parameters."""
        with self.assertRaises(ValueError):
            intrinsic_dimension(self.vector, k1=6, k2=4)

    def test_incorrect_k2_params(self):
        """ Test handling of incorrect neighborhood parameters."""
        n = self.vector.shape[0]
        with self.assertRaises(ValueError):
            intrinsic_dimension(self.vector, k2=n)

    def test_incorrect_trafo_params(self):
        """ Test handling of incorrect transformation parameters."""
        with self.assertRaises(ValueError):
            intrinsic_dimension(self.vector, trafo=0)

    def test_incorrect_metric_dist(self):
        """ Test handling of unsupported metric parameters."""
        with self.assertRaises(NotImplementedError):
            intrinsic_dimension(self.vector, metric='distance')

    def test_incorrect_metric_sim(self):
        """ Test handling of unsupported metric parameters."""
        with self.assertRaises(NotImplementedError):
            intrinsic_dimension(self.vector, metric='similarity')

    def test_incorrect_metric_other(self):
        """ Test handling of unsupported metric parameters."""
        with self.assertRaises(ValueError):
            intrinsic_dimension(self.vector, metric=None)

if __name__ == "__main__":
    unittest.main()
