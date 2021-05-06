__author__ = "Tomasz Rybotycki"

"""
    This file holds the implementation of Ryser method of permanent calculation, as presented in [Cliffords2020].
"""

from typing import List, Optional

from numpy import complex128, ndarray
from scipy.special import binom

from ..permanent_calculators.bs_permanent_calculator_base import BSPermanentCalculatorBase


class RyserPermanentCalculator(BSPermanentCalculatorBase):
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
        for r_vector in r_vectors:
            permanent += self._compute_permanent_addend(r_vector)
        permanent *= pow(-1, sum(self._input_state))
        return permanent

    def _can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self._matrix.shape[0] == self._matrix.shape[1] and len(self._output_state) == len(self._input_state) \
               and len(self._output_state) == self._matrix.shape[0]

    def _calculate_r_vectors(self, input_vector: Optional[ndarray] = None) -> List[ndarray]:
        """
            This method is used to calculate all the r vectors that appear in the Ryser's formula during permanent
            calculation. By r vectors I denote vectors in form [r_1, r_2, ..., r_m]. This method basically takes care
            of the sum of sum ... of sum at the beginning of the formula.
            :return: List of r vectors.
        """
        if input_vector is None:
            input_vector = []
        r_vectors = []
        for i in range(int(self._output_state[len(input_vector)]) + 1):
            input_state = input_vector.copy()
            input_state.append(i)

            if len(input_state) == len(self._output_state):
                r_vectors.append(input_state)
            else:
                r_vectors.extend(self._calculate_r_vectors(input_state))

        return r_vectors

    def _compute_permanent_addend(self, r_vector: ndarray) -> complex128:

        addend = pow(-1, sum(r_vector))

        # Binomials calculation
        for i in range(len(r_vector)):
            addend *= binom(self._output_state[i], r_vector[i])

        # Product calculation
        product = 1
        for j in range(sum(self._input_state)):
            # Otherwise we calculate the sum
            product_part = 0

            for nu in range(len(self._input_state)):
                product_part += r_vector[nu] * self._matrix[nu][j]

            product *= product_part
        addend *= product
        return addend
