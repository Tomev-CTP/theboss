__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Chin and Huhs method of permanent calculation, as presented in [2].
"""

from typing import List, Optional

from numpy import array, asarray, complex128, int64, ndarray
from scipy.special import binom

from src.boson_sampling_utilities.permanent_calculators.BSPermanentCalculatorInterface import \
    BSPermanentCalculatorInterface


class ChinHuhPermanentCalculator(BSPermanentCalculatorInterface):
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are set to [1, 1, ..., 1] with proper dimensions.
    """

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
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """
        if not self._can_calculation_be_performed():
            raise AttributeError

        v_vectors = self._calculate_v_vectors()

        permanent = complex128(0)
        for v_vector in v_vectors:
            permanent += self._compute_permanent_addend(v_vector)
        permanent /= pow(2, sum(self._input_state))
        return permanent

    def _can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self._matrix.shape[0] == self._matrix.shape[1] and len(self._output_state) == len(self._input_state) \
            and len(self._output_state) == self._matrix.shape[0]

    def _calculate_v_vectors(self, input_vector: Optional[ndarray] = None) -> List[ndarray]:
        if input_vector is None:
            input_vector = []
        v_vectors = []
        for i in range(int(self._input_state[len(input_vector)]) + 1):
            input_state = input_vector.copy()
            input_state.append(i)

            if len(input_state) == len(self._input_state):
                v_vectors.append(input_state)
            else:
                v_vectors.extend(self._calculate_v_vectors(input_state))

        return v_vectors

    def _compute_permanent_addend(self, v_vector: ndarray) -> complex128:
        v_sum = sum(v_vector)
        addend = pow(-1, v_sum)
        # Binomials calculation
        for i in range(len(v_vector)):
            addend *= binom(self._input_state[i], v_vector[i])
        # Product calculation
        product = 1
        for j in range(len(self._input_state)):
            if self._output_state[j] == 0:  # There's no reason to calculate the sum if t_j = 0
                continue
            # Otherwise we calculate the sum
            product_part = 0
            for i in range(len(self._input_state)):
                product_part += (self._input_state[i] - 2 * v_vector[i]) * self._matrix[j][i]
            product_part = pow(product_part, self._output_state[j])
            product *= product_part
        addend *= product
        return addend
