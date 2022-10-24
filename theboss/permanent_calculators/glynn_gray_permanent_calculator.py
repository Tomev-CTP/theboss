__author__ = "Tomasz Rybotycki"

"""
    This files contains the implementation of Glynn's formula for permanent calculation,
    that uses Gray codes to during the iterations. This method has the same complexity
    as the Ryser formula (O(n2^n), but has been proven to be numerically more stable
    (due to physical limitations of the machines).
"""

from operator import mul
from functools import reduce

from typing import Optional, List, Sequence

from numpy import ones

from theboss.permanent_calculators.bs_permanent_calculator_base import (
    BSPermanentCalculatorBase,
)
from theboss.boson_sampling_utilities import EffectiveScatteringMatrixCalculator

from guancodes.GrayCode import get_gray_code_update_indices


class GlynnGrayPermanentCalculator(BSPermanentCalculatorBase):
    """
    A matrix permanent calculator that uses the Glynn formula. It first computes the
    effective scattering (reduced) matrix and then iterates through the sum in the
    Glynn formula using the Gray code.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[complex]],
        input_state: Optional[Sequence[int]] = None,
        output_state: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(matrix, input_state, output_state)
        self._delta: Sequence[int]
        self._multiplier: int
        self._sums: List[complex]
        self.permanent: complex
        self._scattering_matrix: Sequence[Sequence[complex]]

    def compute_permanent(self) -> complex:
        """
        This is the method for computing the permanent. It will work as described
        in [0], formula (1), and iterates over deltas in Gray code.

        :return:
            Permanent of the sub-matrix corresponding to the specified input and
            output states.
        """

        # Prepare the matrix.
        self._scattering_matrix = EffectiveScatteringMatrixCalculator(
            self.matrix, self.input_state, self.output_state
        ).calculate()

        if len(self._scattering_matrix) == 0:
            return complex(1)

        self._initialize_permanent_computation()

        update_indices = get_gray_code_update_indices(len(self._scattering_matrix) - 1)

        # Begin the iteration.
        for i in update_indices:
            self._multiplier = -self._multiplier
            self._delta[i] = -self._delta[i]

            for j in range(len(self._sums)):
                self._sums[j] += 2 * self._delta[i] * self._scattering_matrix[i][j]

            self.permanent += self._multiplier * reduce(mul, self._sums, 1)

        self.permanent /= 2 ** (len(self._scattering_matrix) - 1)

        return self.permanent

    def _initialize_permanent_computation(self) -> None:
        """Prepares the class for permanent computation."""
        self._delta = ones(len(self._scattering_matrix), dtype=int)
        self._multiplier = 1
        self._sums = []

        for j in range(len(self._scattering_matrix)):
            self._sums.append(complex(0))
            for i in range(len(self._delta)):
                self._sums[-1] += self._delta[i] * self._scattering_matrix[i][j]

        self.permanent = self._multiplier * reduce(mul, self._sums, 1)
