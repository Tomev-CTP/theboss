__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Chin and Huhs method of permanent calculation, as presented in [2].
"""

from typing import List, Optional

from numpy import complex128, ndarray, nonzero, ones, zeros, allclose, isinf, power
from scipy.special import binom
from functools import reduce
import operator

from ..permanent_calculators.bs_permanent_calculator_base import BSPermanentCalculatorBase


class ChinHuhPermanentCalculator_v2(BSPermanentCalculatorBase):
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are set to [1, 1, ..., 1] with proper dimensions.
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

        # Required for Guan Code iteration
        v_vector = zeros(len(self._input_state), dtype=int)  # g
        code_update_information = ones(len(self._input_state), dtype=int)  # u
        position_limits = list(self._input_state)  # n

        permanent = complex128(0)
        sums = dict()

        binomials_product = 1
        considered_columns_indices = nonzero(self._output_state)[0]

        multiplier = 1

        # Initialization (0-th step).
        for i in considered_columns_indices:
            sums[i] = 0
            for j in range(len(self._input_state)):
                sums[i] += self._input_state[j] * self._matrix[j][i]

        permanent += multiplier * binomials_product * \
                         reduce(operator.mul, [pow(sums[i], self._output_state[i]) for i in considered_columns_indices], 1)

        # Rest of the steps.
        while (v_vector[-1] <= position_limits[-1]):

            # UPDATE R VECTOR
            index_to_update = 0  # i
            updated_value_at_index = v_vector[0] + code_update_information[0]  # k
            while updated_value_at_index > position_limits[
                index_to_update] or updated_value_at_index < 0:
                code_update_information[index_to_update] = -code_update_information[
                    index_to_update]
                index_to_update += 1

                if index_to_update == len(v_vector):

                    for _ in range(sum(self._input_state)):
                        permanent /= 2

                    return permanent

                updated_value_at_index = v_vector[index_to_update] + \
                                         code_update_information[index_to_update]

            last_value_at_index = v_vector[index_to_update]
            v_vector[index_to_update] = updated_value_at_index
            # END UPDATE

            # START PERMANENT UPDATE
            multiplier = -multiplier

            # Sums update
            for i in sums:
                sums[i] -= 2 * (v_vector[index_to_update] - last_value_at_index) * \
                           self.matrix[index_to_update][i]

            # Binoms update
            if v_vector[index_to_update] > last_value_at_index:
                binomials_product *= (self._input_state[index_to_update] -
                                      last_value_at_index) / v_vector[index_to_update]
            else:
                binomials_product *= last_value_at_index / (
                        self._input_state[index_to_update] - v_vector[
                    index_to_update])

            permanent += multiplier * binomials_product * \
                         reduce(operator.mul, [pow(sums[j], self._output_state[j]) for j in considered_columns_indices], 1)

        for _ in range(sum(self._input_state)):
            permanent /= 2

        return permanent

    def _can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self._matrix.shape[0] == self._matrix.shape[1] and len(self._output_state) == len(self._input_state) \
               and len(self._output_state) == self._matrix.shape[0]

    def sums_check(self, v):
        sums = dict()

        for i in range(len(self._output_state)):

            if self._output_state[i] == 0:
                continue

            sums[i] = 0
            for j in range(len(self._input_state)):
                sums[i] += (self._input_state[j] - 2 * v[j]) * self._matrix[j][i]

        return sums
