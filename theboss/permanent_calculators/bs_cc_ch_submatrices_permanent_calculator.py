__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of submatrices calculation method as used
    in Clifford & Clifford version B of the algorithm. C&C algorithm uses Laplace
    expansion for the permanents in order to compute the set of probabilities in each
    step of the algorithm. Instead of computing each permanent separately, we can
    compute them all in one run, which is vastly more efficient.
    
    Instead of using Ryser's formula, or Glynn formula, we base our approach on the
    Chin-Huh's formula, which takes into account possible bunching in the input states
    (or rather can be interpreted as that). 
"""

from numpy import complex128, ndarray, array
import operator
from functools import reduce
from typing import List, Optional

from theboss.permanent_calculators.bs_submatrices_permanent_calculator_base import (
    BSGuanBasedSubmatricesPermanentCalculatorBase,
)


class BSCCCHSubmatricesPermanentCalculator(
    BSGuanBasedSubmatricesPermanentCalculatorBase
):

    """
    The name stands for Boson Sampling Clifford & Clifford Chin-Huh submatrices
    permanent calculator, as it uses Clifford & Clifford approach to compute
    permanents of submatrices to compute sub-distribution of Boson Sampling problem
    instance. The starting point in our case is Chin-Huh permanent calculator
    iterated in Guan Codes induced order.
    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:

        self._sums: dict = dict()
        self.permanents: List[complex128] = []
        self._multiplier: int = 1
        self._considered_columns_indices = array(0)

        super().__init__(matrix, input_state, output_state)

    def _initialize_permanents_computation(self) -> None:
        """
        Initialize all the class fields prior to the permanents computation.
        """
        super()._initialize_permanents_computation()

        self._multiplier = 1 / pow(2, sum(self.input_state) - 1)

        for i in self._considered_columns_indices:
            self._sums[i] = 0
            for j in range(len(self._input_state)):
                self._sums[i] += self._input_state[j] * self._matrix[i][j]

        self._update_permanents()

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

    def _update_permanents(self) -> None:
        """
        Update the intermediate permanents.
        """
        for i in range(len(self.input_state)):

            if self.input_state[i] == 0 or self.input_state[i] == self._r_vector[i]:
                continue

            # Compute update the sums
            updated_sums = {}
            for j in self._considered_columns_indices:
                updated_sums[j] = self._sums[j] - self.matrix[j][i]

            # Compute update binomial product
            updated_binom = self._binomials_product / (
                self.input_state[i] / (self.input_state[i] - self._r_vector[i])
            )

            addend = reduce(
                operator.mul,
                [
                    pow(updated_sums[j], self._output_state[j])
                    for j in self._considered_columns_indices
                ],
                self._multiplier * updated_binom,
            )

            self.permanents[i] += addend
