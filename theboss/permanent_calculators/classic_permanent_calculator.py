__author__ = "Tomasz Rybotycki"

"""
    This class is used to calculate permanent of effective scattering matrix. It first generates the matrix, and then
    calculates the permanent via standard means.
"""

from typing import List, Optional, Sequence

from numpy import asarray, ndarray

from theboss.boson_sampling_utilities import EffectiveScatteringMatrixCalculator
from theboss.permanent_calculators.bs_permanent_calculator_base import (
    BSPermanentCalculatorBase,
)


class ClassicPermanentCalculator(BSPermanentCalculatorBase):
    """
    An implementation of a classic permanent calculator. Classic approach to the BS
    permanent calculation is to compute the sub-matrix (or reduced matrix) according
    to the input and output state for which the permanent is computed, and the compute
    the permanent of the sub-matrix.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[complex]],
        input_state: Optional[Sequence[int]] = None,
        output_state: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__(matrix, input_state, output_state)

    def compute_permanent(self) -> complex:
        """
        Computes permanent of the previously specified matrix. If first computes the
        reduced matrix, and then computes permanent of it.

        :return:
            The permanent of the specified (possibly reduced) matrix.
        """
        if sum(self.input_state) == 0:
            if sum(self.output_state) == 0:
                return complex(1 + 0j)
            else:
                return complex(0)

        scattering_matrix_calculator = EffectiveScatteringMatrixCalculator(
            asarray(self._matrix), self._input_state, self._output_state
        )
        scattering_matrix = scattering_matrix_calculator.calculate()
        return self._compute_permanent_recursively(asarray(scattering_matrix))

    def _compute_permanent_recursively(self, matrix: ndarray) -> complex:
        """
        Returns the permanent of the matrix.
        """
        return self._permanent_recursive_part(
            matrix, column=0, selected=[], prod=complex(1)
        )

    def _permanent_recursive_part(
        self, mtx: ndarray, column: int, selected: List[int], prod: complex
    ) -> complex:
        """
        Row expansion for the permanent of matrix mtx.
        The counter column is the current column,
        selected is a list of indices of selected rows,
        and prod accumulates the current product.
        """
        if column == mtx.shape[1]:
            return prod

        result = complex(0 + 0j)
        for row in range(mtx.shape[0]):
            if row not in selected:
                result += self._permanent_recursive_part(
                    mtx, column + 1, selected + [row], prod * mtx[row, column]
                )
        return result
