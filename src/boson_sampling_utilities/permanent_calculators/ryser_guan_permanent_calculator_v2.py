__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Ryser method of permanent calculation, as presented in [Cliffords2020] with
    using Guan codes to speed it up. I modified it slightly, so that the permanent can be used for computing permanents
    for bunched and not-continuous (like [1, 1, 0, 1, 0]) inputs.
"""

import operator
from functools import reduce
from typing import List, Optional

from numpy import complex128, ndarray, nonzero, zeros, ones

from ..permanent_calculators.bs_permanent_calculator_base import \
    BSPermanentCalculatorBase
from ...GuanCodes.src.GuanCodeGenerator import GuanCodeGenerator


class RyserGuanPermanentCalculator_v2(BSPermanentCalculatorBase):
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

        r_vector = zeros(len(self.input_state), dtype=int)  # g
        code_update_information = ones(len(self.input_state), dtype=int)  # u
        position_limits = list(self.input_state)  # n

        permanent = complex128(0)
        sums = dict()

        considered_columns_indices = nonzero(self._input_state)[0]

        multiplier = 1
        binomials_product = 1

        for j in considered_columns_indices:
            sums[j] = 0

        while (r_vector[-1] <= position_limits[-1]):

            # UPDATE R VECTOR
            index_to_update = 0  # i
            updated_value_at_index = r_vector[0] + code_update_information[0]  # k
            while updated_value_at_index > position_limits[
                index_to_update] or updated_value_at_index < 0:
                code_update_information[index_to_update] = -code_update_information[
                    index_to_update]
                index_to_update += 1

                if index_to_update == len(r_vector):
                    print("May have a problem!")
                    permanent *= pow(-1, sum(self._input_state))
                    return permanent

                updated_value_at_index = r_vector[index_to_update] + \
                                         code_update_information[index_to_update]


            last_value_at_index = r_vector[index_to_update]
            r_vector[index_to_update] = updated_value_at_index
            # END UPDATE

            # START PERMANENT UPDATE
            multiplier = -multiplier

            # Sums update
            for j in sums:
                sums[j] += (r_vector[index_to_update] - last_value_at_index) * \
                           self.matrix[index_to_update][j]

            # Binoms update
            if r_vector[index_to_update] > last_value_at_index:
                binomials_product *= (self._output_state[index_to_update] -
                                      last_value_at_index) / r_vector[index_to_update]
            else:
                binomials_product *= last_value_at_index / (
                        self._output_state[index_to_update] - r_vector[
                    index_to_update])

            permanent += multiplier * binomials_product * reduce(operator.mul,
                                                                 [pow(sums[j],
                                                                      self._input_state[
                                                                          j]) for j in
                                                                  considered_columns_indices],
                                                                 1)

        permanent *= pow(-1, sum(self._input_state))
        return permanent


    def _can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self._matrix.shape[0] == self._matrix.shape[1] and len(
            self._output_state) == len(self._input_state) \
               and len(self._output_state) == self._matrix.shape[0]