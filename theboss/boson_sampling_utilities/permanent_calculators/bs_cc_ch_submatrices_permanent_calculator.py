__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of submatrices calculation method as used
    in Clifford & Clifford version B of the algorithm. C&C algorithm uses Laplace
    expansion for the permanents in order to compute the set of probabilities in each
    step of the algorithm. Instead of computing each permanent separately, we can
    compute them all in one run, which is vastly more efficient.
    
    Instead of using Rysers formula, or Glynn formula, we base our approach on the
    Chin-Huh's formula, which takes into account possible bunching in the input states
    (or rather can be interpreted as that). 
"""

import abc
from numpy import complex128, ndarray, int64, array, nonzero, zeros, ones
import operator
from functools import reduce
from typing import List, Optional


class BSSubmatricesPermanentCalculatorInterface(abc.ABC):
    """
        This is the interface class for submatrices permanents calculator. For now
        BSCCCHSubmatricesPermanentCalculator is the only class of that kind, but it's
        possible that there will be more (based e.g. on Glynns formula that grants
        numerical stability.

        If the above happens, place the interface class in the separate file.
    """

    @abc.abstractmethod
    def compute_permanents(self) -> List[complex128]:
        """Computes permanent of a matrix given before."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def matrix(self) -> ndarray:
        raise NotImplementedError

    @matrix.setter
    @abc.abstractmethod
    def matrix(self, matrix: ndarray) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_state(self) -> ndarray:
        raise NotImplementedError

    @input_state.setter
    @abc.abstractmethod
    def input_state(self, input_state: ndarray) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_state(self) -> ndarray:
        raise NotImplementedError

    @output_state.setter
    @abc.abstractmethod
    def output_state(self, output_state: ndarray) -> None:
        raise NotImplementedError


class BSSubmatricesPermanentCalculatorBase(
    BSSubmatricesPermanentCalculatorInterface, abc.ABC
):
    """
        Base class for BSSubmatricesPermanentCalculator classes. It takes care of some
        boilerplate code.

        Again, it should be put into separate file were the
        BSCCCHSubmatricesPermanentCalculator cease to be the only submatrices
        permanent calculator.
    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:
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
        self._input_state = array(input_state, dtype=int64)

    @property
    def output_state(self) -> ndarray:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self._output_state = array(output_state, dtype=int64)


class BSCCCHSubmatricesPermanentCalculator(BSSubmatricesPermanentCalculatorBase):

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

        self.sums: dict = dict()
        self.permanents: List[complex128] = []
        self.multiplier: int = 1
        self.binomials_product: int = 1
        self.v_vector: ndarray = array(0)
        self.considered_columns_indices = array(0)

        super().__init__(matrix, input_state, output_state)

    def compute_permanents(self) -> List[complex128]:

        # TODO TR:  This method is huge and complicated. It would be smart to break
        #           it down into smaller ones.

        self.permanents = [complex128(0) for _ in range(len(self.input_state))]

        # Required for Guan Code iteration
        self.v_vector = zeros(len(self._input_state), dtype=int)  # g
        code_update_information = ones(len(self._input_state), dtype=int)  # u
        position_limits = list(self._input_state)  # n

        self.sums = dict()
        self.binomials_product = 1
        self.considered_columns_indices = nonzero(self._output_state)[0]

        self.multiplier = 1

        # Initialization (0-th step).
        for i in self.considered_columns_indices:
            self.sums[i] = 0
            for j in range(len(self._input_state)):
                self.sums[i] += self._input_state[j] * self._matrix[i][j]

        self._add_permanent_addends()

        # Rest of the steps.
        while self.v_vector[-1] <= position_limits[-1]:

            # UPDATE R VECTOR
            index_to_update = 0  # i
            updated_value_at_index = self.v_vector[0] + code_update_information[0]  # k
            while (
                updated_value_at_index > position_limits[index_to_update]
                or updated_value_at_index < 0
            ):
                code_update_information[index_to_update] = -code_update_information[
                    index_to_update
                ]
                index_to_update += 1

                if index_to_update == len(self.v_vector):

                    for _ in range(int(sum(self.input_state)) - 1):
                        for i in range(len(self.permanents)):
                            self.permanents[i] /= 2

                    return self.permanents

                updated_value_at_index = (
                    self.v_vector[index_to_update]
                    + code_update_information[index_to_update]
                )

            last_value_at_index = self.v_vector[index_to_update]
            self.v_vector[index_to_update] = updated_value_at_index
            # END UPDATE

            # START PERMANENT UPDATE
            self.multiplier = -self.multiplier

            # Sums update
            for i in self.sums:
                self.sums[i] -= (
                    2
                    * (self.v_vector[index_to_update] - last_value_at_index)
                    * self.matrix[i][index_to_update]
                )

            # Binoms update
            if self.v_vector[index_to_update] > last_value_at_index:
                self.binomials_product *= (
                    self._input_state[index_to_update] - last_value_at_index
                ) / self.v_vector[index_to_update]
            else:
                self.binomials_product *= last_value_at_index / (
                    self._input_state[index_to_update] - self.v_vector[index_to_update]
                )

            self._add_permanent_addends()

        for _ in range(int(sum(self._input_state)) - 1):
            for i in range(len(self.permanents)):
                self.permanents[i] /= 2

        return self.permanents

    def _add_permanent_addends(self) -> None:
        # For each occupied mode
        for i in range(len(self.input_state)):

            if self.input_state[i] == 0 or self.input_state[i] == self.v_vector[i]:
                continue

            # Compute update the sums
            updated_sums = {}
            for j in self.considered_columns_indices:
                updated_sums[j] = self.sums[j] - self.matrix[j][i]

            # Compute update binomial product
            updated_binom = self.binomials_product / (
                self.input_state[i] / (self.input_state[i] - self.v_vector[i])
            )

            addend = (
                self.multiplier
                * updated_binom
                * reduce(
                    operator.mul,
                    [
                        pow(updated_sums[j], self._output_state[j])
                        for j in self.considered_columns_indices
                    ],
                    1,
                )
            )

            self.permanents[i] += addend
