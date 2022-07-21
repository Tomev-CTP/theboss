__author__ = "Tomasz Rybotycki"

"""
    This script contains tests for the boson sampling utilities.
"""

import unittest


# TODO TR: Add 1st Q -> 2nd Q -> 1st Q test, and vice 2 -> 1 -> 2 test.
# TODO TR: Test if the number of possible n-mode m-particle states are correct.

class TestQuantumComputationsUtilities(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix_size = 5
        self.number_of_matrices_for_distinct_elements_check = 10  # Should be >= 2.

    def test_something(self) -> None:
        self.assertTrue(False)