__author__ = "Tomasz Rybotycki"

"""
    This class is used to calculate permanent of effective scattering matrix. It first generates the matrix, and then
    calculates the permanent via standard means.
"""

from typing import List, Optional

from numpy import array, asarray, complex128, int64, ndarray

from src.boson_sampling_utilities.Boson_Sampling_Utilities import EffectiveScatteringMatrixCalculator
from src.boson_sampling_utilities.permanent_calculators.BSPermanentCalculatorInterface import \
    BSPermanentCalculatorInterface


class ClassicPermanentCalculator(BSPermanentCalculatorInterface):

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        if output_state is None:
            output_state = array([], dtype=int64)
        if input_state is None:
            input_state = array([], dtype=int64)
        self._matrix = matrix
        self._input_state = input_state
        self._output_state = output_state

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: ndarray) -> None:
        self._matrix = matrix

    @property
    def input_state(self) -> ndarray:
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: ndarray) -> None:
        self._input_state = asarray(input_state, dtype=int64)

    @property
    def output_state(self) -> ndarray:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self._output_state = asarray(output_state, dtype=int64)

    def compute_permanent(self) -> complex128:
        scattering_matrix_calculator = \
            EffectiveScatteringMatrixCalculator(self._matrix, self._input_state, self._output_state)
        scattering_matrix = scattering_matrix_calculator.calculate()
        return self._compute_permanent_recursively(scattering_matrix)

    def _compute_permanent_recursively(self, matrix: ndarray) -> complex128:
        """
        Returns the permanent of the matrix.
        """
        return self._permanent_recursive_part(matrix, column=0, selected=[], prod=complex128(1))

    def _permanent_recursive_part(self, mtx: ndarray, column: int, selected: List[int], prod: complex128) -> complex128:
        """
        Row expansion for the permanent of matrix mtx.
        The counter column is the current column,
        selected is a list of indices of selected rows,
        and prod accumulates the current product.
        """
        if column == mtx.shape[1]:
            return prod

        result = complex128(0 + 0j)
        for row in range(mtx.shape[0]):
            if row not in selected:
                result += self._permanent_recursive_part(mtx, column + 1, selected + [row], prod * mtx[row, column])
        return result
