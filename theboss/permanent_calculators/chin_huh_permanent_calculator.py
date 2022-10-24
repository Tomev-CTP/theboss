__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Chin and Huhs method of permanent calculation,
    as presented in [2]. We use Guan codes to speed up the computations.
"""


from typing import Optional, Dict, Sequence

from theboss.permanent_calculators.bs_permanent_calculator_base import (
    BSGuanCodeBasedPermanentCalculatorBase,
)


class ChinHuhPermanentCalculator(BSGuanCodeBasedPermanentCalculatorBase):
    """
    This class is designed to compute the permanent of the effective scattering matrix
    in a BosonSampling experiment instance using Chin & Huh's method.

    .. note::
        It can be used to compute the permanent of any given matrix. All that is
        required that input and output states are set to :math:`[1, 1, ..., 1]` with
        proper dimensions.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[complex]],
        input_state: Optional[Sequence[int]] = None,
        output_state: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(matrix, input_state, output_state)
        self._multiplier: int
        self._considered_columns_indices: Sequence[int]
        self.permanent: complex
        self._sums: Dict[int, complex]

    def _initialize_permanent_computation(self) -> None:

        super()._initialize_permanent_computation()
        self._multiplier = 1 / pow(2, sum(self._input_state))

        for i in self._considered_columns_indices:
            self._sums[i] = 0
            for j in range(len(self._input_state)):
                self._sums[i] += self._input_state[j] * self._matrix[i][j]

        self._update_permanent()

    def _update_sums(self) -> None:
        """
        Update sums instead of recomputing them.
        """
        for i in self._sums:
            self._sums[i] -= (
                2
                * (self._r_vector[self._index_to_update] - self._last_value_at_index)
                * self.matrix[i][self._index_to_update]
            )
