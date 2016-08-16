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
from hub_toolbox.IO import load_dexter
from hub_toolbox.IntrinsicDim import intrinsic_dimension

class TestIntrinsicDim(unittest.TestCase):

    def setUp(self):
        _, _, self.vector = load_dexter()

    def tearDown(self):
        del self.vector

    def test_intrinsic_dim_mle_levina(self):
        """Test against value calc. by matlab reference implementation."""
        ID_MLE_REF = 74.742
        id_mle = intrinsic_dimension(self.vector, k1=6, k2=12, 
            estimator='levina', metric='vector', trafo=None)
        return self.assertEqual(id_mle, int(ID_MLE_REF))

if __name__ == "__main__":
    unittest.main()
