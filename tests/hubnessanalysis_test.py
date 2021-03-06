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
from hub_toolbox import hubness_analysis
from hub_toolbox.distances import euclidean_distance

class TestHubnessAnalysis(unittest.TestCase):
    """Test the hubness_analysis class (check for results,
       but not for *correct* results.)
    """

    def setUp(self):
        points = 100
        dim = 10
        self.vector = 99. * (np.random.rand(points, dim) - 0.5)
        self.label = np.random.randint(0, 5, points)
        self.dist = euclidean_distance(self.vector)
        self.SEC_DIST = set(['mp', 'mp_gaussi', 'mp_gammai', 
                            'ls', 'nicdm', 'snn', 'cent', 'wcent', 'lcent', 
                            'dsg', 'dsl', 'orig'])

    def tearDown(self):
        del self.dist, self.label, self.vector, self.SEC_DIST

    def test_all_sec_dist_are_covered_in_unittests(self):
        n_self_sec_dist = len(self.SEC_DIST)
        hub_ana_sec_dist = set(hubness_analysis.SEC_DIST.keys())
        n_intersection = len(hub_ana_sec_dist & self.SEC_DIST)
        return self.assertEqual(n_self_sec_dist, n_intersection)

    def test_all_sec_dist_have_header(self):
        ha_sec_dist = set(hubness_analysis.SEC_DIST.keys())
        header_sec_dist = set(hubness_analysis.HubnessAnalysis()._header.keys())
        n_sec_dist = len(ha_sec_dist)
        n_intersection = len(ha_sec_dist & header_sec_dist)
        return self.assertEqual(n_sec_dist, n_intersection)

    def test_all_sec_dist_types(self):
        got_all_results = True
        for dist_type in self.SEC_DIST:
            got_all_results &= self._perform(dist_type)
        return self.assertTrue(got_all_results)

    def _perform(self, dist_type):
        """Test whether the given secondary distance type is supported."""
        ana = hubness_analysis.HubnessAnalysis(
            self.dist, self.label, self.vector, 'distance')
        ana = ana.analyze_hubness(
            experiments=dist_type, print_results=True, verbose=1)
        exp = ana.experiments[0]
        got_all_results = \
            (exp.secondary_distance is not None and
             len(exp.hubness) > 0 and
             len(exp.anti_hubs) > 0 and
             len(exp.max_hub_k_occurence) > 0 and
             len(exp.knn_accuracy) > 0 and
             exp.gk_index is not None and
             ana.intrinsic_dim is not None)
        return got_all_results

    def test_hubness_analysis_only_with_distances(self):
        """ Check correct handling when no labels, vectors are given."""
        ana = hubness_analysis.HubnessAnalysis(self.dist)
        ana = ana.analyze_hubness("orig")
        exp = ana.experiments[0]
        got_all_results = \
            (exp.secondary_distance is not None and
             len(exp.hubness) > 0 and
             len(exp.anti_hubs) > 0 and
             len(exp.max_hub_k_occurence) > 0 and
             exp.gk_index is not None and
             ana.intrinsic_dim is not None)
        return got_all_results

if __name__ == "__main__":
    unittest.main()
