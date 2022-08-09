__author__ = "Tomasz Rybotycki"

"""
    This script contains the tests for submatrices permanent calculators.
"""

import unittest
from theboss.permanent_calculators.bs_cc_ch_submatrices_permanent_calculator import (
    BSCCCHSubmatricesPermanentCalculator,
)
from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)
from theboss.permanent_calculators.bs_cc_ryser_submatrices_permanent_calculator import (
    BSCCRyserSubmatricesPermanentCalculator,
)
from theboss.permanent_calculators.bs_submatrices_permanent_calculator_base import (
    BSSubmatricesPermanentCalculatorBase,
)

from scipy.stats import unitary_group
from numpy import ones, zeros, ndarray

from typing import Callable, Optional


class TestSubmatricesPermanentsCalculators(unittest.TestCase):
    def setUp(self) -> None:

        self._dim = 4

        self._matrix = unitary_group.rvs(self._dim)

        self._input_state = ones(self._dim, dtype=int)

        self._output_state = ones(self._dim, dtype=int)
        self._output_state[0] = 0

        self._permanent_calculator = RyserPermanentCalculator(self._matrix)

    def test_chin_huh_submatrices_permanent_calculator_for_std_input(self) -> None:
        self._input_state = ones(self._dim, dtype=int)
        self._compute_permanents_and_assert(BSCCCHSubmatricesPermanentCalculator)

    def test_chin_huh_submatrices_permanent_calculator_for_collision_input(
        self,
    ) -> None:
        self._input_state = zeros(self._dim)
        self._input_state[0] = 1
        self._input_state[1] = self._dim - 1
        self._compute_permanents_and_assert(BSCCCHSubmatricesPermanentCalculator)

    def test_chin_huh_submatrices_permanent_calculator_for_edge_case(self) -> None:
        self._input_state = zeros(self._dim)
        self._input_state[0] = 1
        self._output_state = zeros(self._dim)
        self._compute_permanents_and_assert(BSCCCHSubmatricesPermanentCalculator)

    def test_ryser_submatrices_permanent_calculator_for_std_input(self) -> None:
        self._input_state = ones(self._dim, dtype=int)
        self._compute_permanents_and_assert(BSCCRyserSubmatricesPermanentCalculator)

    def test_ryser_submatrices_permanent_calculator_for_collision_input(self) -> None:
        self._input_state = zeros(self._dim)
        self._input_state[0] = 1
        self._input_state[1] = self._dim - 1
        self._compute_permanents_and_assert(BSCCRyserSubmatricesPermanentCalculator)

    def test_ryser_submatrices_permanent_calculator_for_edge_case(self) -> None:
        self._input_state = zeros(self._dim)
        self._input_state[0] = 1
        self._output_state = zeros(self._dim)
        self._compute_permanents_and_assert(BSCCRyserSubmatricesPermanentCalculator)

    def _compute_permanents_and_assert(
        self,
        calculator_constructor: Callable[
            [ndarray, Optional[ndarray], Optional[ndarray]],
            BSSubmatricesPermanentCalculatorBase,
        ],
    ) -> None:

        permanents_calculator = calculator_constructor(
            self._matrix, self._input_state, self._output_state
        )

        submatrices_permanents_all = permanents_calculator.compute_permanents()

        submatrices_permanents_single = []
        self._permanent_calculator.output_state = self._output_state

        for i in range(len(self._input_state)):
            self._input_state[i] -= 1

            if self._input_state[i] < 0:
                submatrices_permanents_single.append(0)
            else:
                self._permanent_calculator.input_state = self._input_state
                submatrices_permanents_single.append(
                    self._permanent_calculator.compute_permanent()
                )

            self._input_state[i] += 1

        for i in range(len(submatrices_permanents_all)):
            self.assertAlmostEqual(
                abs(submatrices_permanents_all[i]),
                abs(submatrices_permanents_single[i]),
            )
