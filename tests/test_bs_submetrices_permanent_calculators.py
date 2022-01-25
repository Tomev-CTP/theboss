__author__ = "Tomasz Rybotycki"

"""
    This script contains the tests for submatrices permanent calculators.
"""

import unittest
from ..theboss.boson_sampling_utilities.permanent_calculators.bs_cc_ch_submatrices_permanent_calculator import BSCCCHSubmatricesPermanentCalculator
from ..theboss.boson_sampling_utilities.permanent_calculators.chin_huh_permanent_calculator import ChinHuhPermanentCalculator
from ..theboss.quantum_computations_utilities import generate_haar_random_unitary_matrix
from numpy import ones


class TestSubmatricesPermanentsCalculators(unittest.TestCase):

    def setUp(self) -> None:

        dim = 4

        self._matrix = generate_haar_random_unitary_matrix(dim)
        self._input_state = ones(dim, dtype=int)
        self._output_state = ones(dim, dtype=int)
        self._output_state[0] = 0

        self._submatrices_permanents_calculator = BSCCCHSubmatricesPermanentCalculator(
            self._matrix, self._input_state, self._output_state
        )

        self._permanent_calculator = ChinHuhPermanentCalculator(self._matrix)

    def test_submatrices_permanent_calculator(self) -> None:
        submatrices_permanents_all = self._submatrices_permanents_calculator.compute_permanents()

        submatrices_permanents_single = []
        self._permanent_calculator.output_state = self._output_state

        for i in range(len(self._input_state)):
            self._input_state[i] -= 1

            self._permanent_calculator.input_state = self._input_state
            submatrices_permanents_single.append(self._permanent_calculator.compute_permanent())

            self._input_state[i] += 1

        for i in range(len(submatrices_permanents_all)):
            self.assertAlmostEqual(abs(submatrices_permanents_all[i]),
                                   abs(submatrices_permanents_single[i]))