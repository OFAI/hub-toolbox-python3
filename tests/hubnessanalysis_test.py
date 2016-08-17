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
from hub_toolbox import HubnessAnalysis
from hub_toolbox.Distances import euclidean_distance

class ParametrizedTestCase(unittest.TestCase):
    """ TestCase classes that you want to be parametrized should
        inherit from this class (© 2003-2016 Eli Bendersky, unlicense,
        http://eli.thegreenplace.net/pages/code)
    """
    def __init__(self, methodName='runTest', param=None):
        super(ParametrizedTestCase, self).__init__(methodName)
        self.param = param

    @staticmethod
    def parametrize(testcase_klass, param=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameter 'param'.
        """
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(testcase_klass)
        suite = unittest.TestSuite()
        for name in testnames:
            suite.addTest(testcase_klass(name, param=param))
        return suite

class TestHubnessAnalysis(ParametrizedTestCase):
    """Test the HubnessAnalysis class (check for results,
       but not for *correct* results.
    """

    def setUp(self):
        points = 100
        dim = 10
        self.vector = 99. * (np.random.rand(points, dim) - 0.5)
        self.label = np.random.randint(0, 5, points)
        self.dist = euclidean_distance(self.vector)

    def tearDown(self):
        del self.dist, self.label, self.vector

    def test_dist_type(self):
        """Test whether the given secondary distance type is supported."""
        ana = HubnessAnalysis.HubnessAnalysis(
            self.dist, self.label, self.vector, 'distance')
        ana = ana.analyze_hubness(
            experiments=self.param, print_results=False)
        exp = ana.experiments[0]
        got_all_results = \
            (exp.secondary_distance is not None and
             len(exp.hubness) > 0 and
             len(exp.anti_hubs) > 0 and
             len(exp.max_hub_k_occurence) > 0 and
             len(exp.knn_accuracy) > 0 and
             exp.gk_index is not None and
             ana.intrinsic_dim is not None)
        return self.assertTrue(got_all_results)

if __name__ == "__main__":
    hub_test_suite = unittest.TestSuite()
    for dist_type in HubnessAnalysis.SEC_DIST.keys():
        hub_test_suite.addTest(ParametrizedTestCase.parametrize(
            TestHubnessAnalysis, param=dist_type))
    unittest.TextTestRunner(verbosity=1).run(hub_test_suite)
