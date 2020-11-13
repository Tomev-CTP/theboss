__author__ = 'Tomasz Rybotycki'

"""
Throughout the Boson Sampling Simulation methods there are many different methods of calculating effective scattering
matrix permanent (that is related to samples probabilities). Given that in this project there already are more than 1
of them implemented, we'd like to ensure, that all of them give the same results.

For test purposes I assume that number of observed modes is m = 4 and number of bosons n varies in the tests.
Note that increasing or decreasing m will require adequate updates of input and output states in test cases. 
"""

import unittest
from sys import float_info
from typing import List

from numpy import array, complex128

from src.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, EffectiveScatteringMatrixPermanentCalculator
from src.Quantum_Computations_Utilities import generate_haar_random_unitary_matrix


class TestEffectiveScatteringMatrixPermanentsCalculators(unittest.TestCase):

    def setUp(self) -> None:
        self._matrix = generate_haar_random_unitary_matrix(4)
        self._ch_permanent_calculator = ChinHuhPermanentCalculator(self._matrix, array([]), array([]))
        self._cl_permanent_calculator = EffectiveScatteringMatrixPermanentCalculator(self._matrix, array([]), array([]))
        pass

    @staticmethod
    def __complex_almost_equal(a: complex128, b: complex128) -> bool:
        difference = a - b
        return abs(difference) < float_info.epsilon

    def __set_input_and_output_states(self, input_state: List[int], output_state: List[int]) -> None:
        self._ch_permanent_calculator.input_state = input_state
        self._ch_permanent_calculator.output_state = output_state
        self._cl_permanent_calculator.input_state = input_state
        self._cl_permanent_calculator.output_state = output_state

    def test_full_input_output_case(self) -> None:
        self.__set_input_and_output_states([1, 1, 1, 1], [1, 1, 1, 1])
        self.assertTrue(self.__complex_almost_equal(self._cl_permanent_calculator.calculate(),
                                                    self._ch_permanent_calculator.calculate()))

    def test_not_full_input_output_case(self) -> None:
        self.__set_input_and_output_states([1, 1, 1, 0], [1, 0, 1, 1])
        self.assertTrue(self.__complex_almost_equal(self._cl_permanent_calculator.calculate(),
                                                    self._ch_permanent_calculator.calculate()))

    def test_binned_input_case(self) -> None:
        self.__set_input_and_output_states([2, 1, 0, 0], [0, 1, 1, 1])
        self.assertTrue(self.__complex_almost_equal(self._cl_permanent_calculator.calculate(),
                                                    self._ch_permanent_calculator.calculate()))

    def test_binned_output_case(self) -> None:
        self.__set_input_and_output_states([1, 1, 1, 0], [2, 1, 0, 0])
        self.assertTrue(self.__complex_almost_equal(self._cl_permanent_calculator.calculate(),
                                                    self._ch_permanent_calculator.calculate()))

    def test_binned_input_binned_output_case(self) -> None:
        self.__set_input_and_output_states([2, 1, 1, 0], [1, 1, 0, 2])
        self.assertTrue(self.__complex_almost_equal(self._cl_permanent_calculator.calculate(),
                                                    self._ch_permanent_calculator.calculate()))
