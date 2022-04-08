__author__ = "Tomasz Rybotycki"

"""
    This files contains the implementation of Glynn's formula for permanent calculation,
    that uses Gray codes to during the iterations. This method has the same complexity
    as the Ryser formula (O(n2^n), but has been proven to be numerically more stable
    (due to physical limitations of the machines).
"""

try:
    from math import prod
except:
    def prod(values):

        if len(values) == 0:
            return 0

        prod = 1

        for v in values:
            prod *= v

        return prod

from typing import Optional

from numpy import ndarray, complex128, ones

from .bs_permanent_calculator_base import BSPermanentCalculatorBase
from ..boson_sampling_utilities import EffectiveScatteringMatrixCalculator
from guancodes.GrayCode import get_gray_code_update_indices


class GlynnGrayPermanentCalculator(BSPermanentCalculatorBase):

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        super().__init__(matrix, input_state, output_state)

    def compute_permanent(self) -> complex128:
        """
            This is the method for computing the permanent. It will work as described
            in [0], meaning implement formula (1) and iterate over deltas in Gray code.
        """

        # Prepare the matrix.
        self.matrix = EffectiveScatteringMatrixCalculator(self.matrix, self.input_state,
                                                          self.output_state).calculate()

        if len(self.matrix) == 0:
            return complex128(1)

        # Initialize the variables.
        delta = ones(len(self.matrix), dtype=int)
        multiplier = 1
        sums = []

        for j in range(len(self.matrix)):
            sums.append(complex128(0))
            for i in range(len(delta)):
                sums[-1] += delta[i] * self.matrix[i][j]

        permanent = multiplier * prod(sums)

        update_indices = get_gray_code_update_indices(len(self.matrix) - 1)

        # Begin the iteration.
        for i in update_indices:
            multiplier = -multiplier
            delta[i] = -delta[i]

            # Sums update
            for j in range(len(sums)):
                sums[j] += 2 * delta[i] * self.matrix[i][j]

            permanent += multiplier * prod(sums)

        return permanent / 2 ** (len(self.matrix) - 1)
