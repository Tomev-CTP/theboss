__author__ = "Tomasz Rybotycki"

"""
Throughout the Boson Sampling Simulation methods there are many different methods of calculating effective scattering
matrix permanent (that is related to samples probabilities). Given that in this project there already are more than 1
of them implemented, we'd like to ensure, that all of them give the same results.

For test purposes I assume that number of observed modes is m = 4 and number of bosons n varies in the tests.
Note that increasing or decreasing m will require adequate updates of input and output states in test cases. 
"""

import unittest
from typing import List

from numpy import allclose, array

from src_tests import (ClassicPermanentCalculator,
                       ChinHuhPermanentCalculator,
                       ParallelChinHuhPermanentCalculator,
                       RyserPermanentCalculator,
                       RyserGuanPermanentCalculator)
from src_tests import generate_haar_random_unitary_matrix


class TestEffectiveScatteringMatrixPermanentsCalculators(unittest.TestCase):

    def setUp(self) -> None:
        self._matrix = generate_haar_random_unitary_matrix(4)

        self._cl_permanent_calculator = ClassicPermanentCalculator(
            matrix=self._matrix, input_state=array([]), output_state=array([])
        )

        self._ch_v2_permanent_calculator = ChinHuhPermanentCalculator(
            matrix=self._matrix, input_state=array([]), output_state=array([])
        )

        self._pch_permanent_calculator = ParallelChinHuhPermanentCalculator(
            matrix=self._matrix, input_state=array([]), output_state=array([])
        )

        self._r_permanent_calculator = RyserPermanentCalculator(
            matrix=self._matrix, input_state=array([]), output_state=array([])
        )

        self._rg_v2_permanent_calculator = RyserGuanPermanentCalculator(
            matrix=self._matrix, input_state=array([]), output_state=array([])
        )

    def _set_input_and_output_states(self, input_state: List[int],
                                     output_state: List[int]) -> None:
        self._cl_permanent_calculator.input_state = input_state
        self._cl_permanent_calculator.output_state = output_state

        self._ch_v2_permanent_calculator.input_state = input_state
        self._ch_v2_permanent_calculator.output_state = output_state

        self._pch_permanent_calculator.input_state = input_state
        self._pch_permanent_calculator.output_state = output_state

        self._r_permanent_calculator.input_state = input_state
        self._r_permanent_calculator.output_state = output_state

        self._rg_v2_permanent_calculator.input_state = input_state
        self._rg_v2_permanent_calculator.output_state = output_state

    def _do_all_assertions(self) -> None:
        cl_permanent = self._cl_permanent_calculator.compute_permanent()

        self.assertTrue(allclose(cl_permanent,
                                 self._ch_v2_permanent_calculator.compute_permanent()))

        self.assertTrue(
            allclose(cl_permanent, self._r_permanent_calculator.compute_permanent()))

        self.assertTrue(allclose(cl_permanent,
                                 self._rg_v2_permanent_calculator.compute_permanent()))

    def test_full_input_output_case(self) -> None:
        self._set_input_and_output_states([1, 1, 1, 1], [1, 1, 1, 1])
        self._do_all_assertions()

    def test_not_full_input_output_case(self) -> None:
        self._set_input_and_output_states([1, 1, 1, 0], [1, 0, 1, 1])
        self._do_all_assertions()

    def test_binned_input_case(self) -> None:
        self._set_input_and_output_states([2, 1, 0, 0], [0, 1, 1, 1])
        self._do_all_assertions()

    def test_binned_output_case(self) -> None:
        self._set_input_and_output_states([1, 1, 1, 0], [2, 1, 0, 0])
        self._do_all_assertions()

    def test_binned_input_binned_output_case(self) -> None:
        self._set_input_and_output_states([2, 1, 1, 0], [1, 1, 0, 2])
        self._do_all_assertions()

    def test_skipping_input_case(self) -> None:
        self._set_input_and_output_states([0, 0, 4, 0], [1, 1, 0, 2])
        self._do_all_assertions()

    def test_skipping_output_case(self) -> None:
        self._set_input_and_output_states([1, 1, 0, 2], [0, 1, 3, 0])
        self._do_all_assertions()

    def test_skipping_input_skipping_output_case(self) -> None:
        self._set_input_and_output_states([0, 2, 0, 2], [0, 0, 4, 0])
        self._do_all_assertions()

    def test_edge_case_no_particles(self) -> None:
        self._set_input_and_output_states([0, 0, 0, 0], [0, 0, 0, 0])
        self._do_all_assertions()

    """
        TR: Given that the only difference between Chin-Huh and Parallel Chin-Huh is the parallelization in the main
            method, the only thing I believe I need to check is if it works. Results are expected to be the same in
            every case. 
    """

    def test_parallel_binned_input_binned_output_case(self) -> None:
        self._set_input_and_output_states([2, 1, 1, 0], [1, 1, 0, 2])
        self.assertTrue(allclose(self._cl_permanent_calculator.compute_permanent(),
                                 self._pch_permanent_calculator.compute_permanent()))
