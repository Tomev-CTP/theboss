__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Ryser method of permanent calculation, as
    presented in [5] with using Guan codes to speed it up. I modified it slightly, so
    that the permanent can be used for computing permanents for bunched and 
    not-continuous (like [1, 1, 0, 1, 0]) inputs.
"""

from typing import Optional, Dict

from numpy import complex128, ndarray

from ..permanent_calculators.bs_permanent_calculator_base import (
    BSGuanCodeBasedPermanentCalculatorBase,
)


class RyserPermanentCalculator(BSGuanCodeBasedPermanentCalculatorBase):
    """
    This class is designed to calculate permanent of effective scattering matrix of
    a boson sampling instance. Note, that it can be used to calculate permanent of
    a given matrix. All that is required that input and output states are correct
    representations of Fock states with proper dimensions.

    An important thing to note is that in order to use general approach provided by
    the boilerplate code in BSGuanCodeBasedPermanentCalculatorBase, we've got to
    change the role of input and output states. This can be done if we transpose the
    matrix. Look into BSCCRyserSubmatricesPermanentCalculator class description
    for more details.
    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:
        super().__init__(matrix, input_state, output_state)
        self._multiplier: int
        self._considered_columns_indices: ndarray
        self.permanent: complex128
        self._sums: Dict[int, complex128]
        self._matrix_t: ndarray = matrix.transpose()

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: ndarray) -> None:
        self._matrix = matrix
        self._matrix_t = matrix.transpose()

    def _initialize_permanent_computation(self) -> None:
        """Prepares the calculator for permanent computation."""
        super()._initialize_permanent_computation()
        self._multiplier = pow(-1, sum(self._input_state))

        for j in self._considered_columns_indices:
            self._sums[j] = 0

        self._update_permanent()

    def _update_sums(self) -> None:
        """
        Update sums instead of recomputing them.
        """
        for j in self._sums:
            self._sums[j] += (
                self._r_vector[self._index_to_update] - self._last_value_at_index
            ) * self._matrix_t[self._index_to_update][j]

    def _return_permanent(self) -> complex128:
        """
        Given that the (-1)^n factor is taken care in the self._multiplier
        initialization no other operations are required when returning the permanent.
        """
        return self.permanent
