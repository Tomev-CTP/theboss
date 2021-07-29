__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Ryser method of permanent calculation, as presented in [Cliffords2020] with
    using Guan codes to speed it up. I modified it slightly, so that the permanent can be used for computing permanents
    for bunched and not-continuous (like [1, 1, 0, 1, 0]) inputs.
"""

import operator
from functools import reduce
from typing import List, Optional

from numpy import complex128, ndarray, nonzero

from ..permanent_calculators.bs_permanent_calculator_base import \
    BSPermanentCalculatorBase
from ...GuanCodes.src.GuanCodeGenerator import GuanCodeGenerator


class RyserGuanPermanentCalculator(BSPermanentCalculatorBase):
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are correct representations of Fock states with proper dimensions.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        super().__init__(matrix, input_state, output_state)

    def compute_permanent(self) -> complex128:
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """
        if not self._can_calculation_be_performed():
            raise AttributeError

        r_vectors = self._calculate_r_vectors()

        permanent = complex128(0)
        sums = dict()

        considered_columns_indices = nonzero(self._input_state)[0]

        multiplier = 1
        binomials_product = 1

        for i in range(len(r_vectors)):
            r_vector = r_vectors[i]

            if i == 0:  # Initialize sums
                for j in considered_columns_indices:
                    right_sum = 0

                    for nu in range(len(self._input_state)):
                        right_sum += r_vector[nu] * self._matrix[nu][j]

                    sums[j] = right_sum
            else:  # Update binomial product and sum
                multiplier = -multiplier
                last_r_vector = r_vectors[i - 1]
                change_index = self._find_change_index(r_vector, last_r_vector)

                # Sums update
                for j in sums:
                    sums[j] += (r_vector[change_index] - last_r_vector[change_index]) * \
                               self.matrix[change_index][j]

                # Binoms update
                if r_vector[change_index] > last_r_vector[change_index]:
                    binomials_product *= (self._output_state[change_index] -
                                          last_r_vector[change_index]) / r_vector[
                                             change_index]
                else:
                    binomials_product *= last_r_vector[change_index] / (
                            self._output_state[change_index] - r_vector[
                        change_index])

            permanent += multiplier * binomials_product * reduce(operator.mul,
                                                                 [pow(sums[j],
                                                                      self._input_state[
                                                                          j]) for j in
                                                                  considered_columns_indices],
                                                                 1)

        permanent *= pow(-1, sum(self._input_state))
        return permanent

    def _find_change_index(self, r_vector: List[int], last_r_vector: List[int]) -> int:
        for index, (first, second) in enumerate(zip(r_vector, last_r_vector)):
            if first != second:
                return index  # Guan codes assure that the change occurs on only one index

    def _can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self._matrix.shape[0] == self._matrix.shape[1] and len(
            self._output_state) == len(self._input_state) \
               and len(self._output_state) == self._matrix.shape[0]

    def _calculate_r_vectors(self) -> List[List[int]]:
        return GuanCodeGenerator.generate_guan_codes(self._output_state)
