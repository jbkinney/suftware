"""
Unit tests for the density class
"""

from unittest import TestCase

import sys
sys.path.append('../src')

import numpy as np
import unittest
import  suftware as sw
import os

class Density1d(TestCase):

    def setUp(self):
        self.N = 5
        self.data = sw.simulate_density_data(distribution_type='uniform', N=self.N,seed=1)

    def tearDown(self):
        pass

    # method that checks the main calculation of deft_1d by calling _run and ensuring that we get the correct Q_star
    def test_density(self):
        actual_Q_star = Q = sw.DensityEstimator(self.data)
        expected_Q_star = np.array([.56458204,  1.66943372,  1.56915093,  1.29922676,  0.94761056,  0.60883489, 0.34458301])
        self.assertEqual(actual_Q_star.Q_star.evaluate(actual_Q_star.grid).all(),expected_Q_star.all())

    # helper method for test_get_data_file_hand()
    def raiseFileNotFoundError(self):
        return FileNotFoundError

suite = unittest.TestLoader().loadTestsFromTestCase(Density1d)
unittest.TextTestRunner(verbosity=2).run(suite)