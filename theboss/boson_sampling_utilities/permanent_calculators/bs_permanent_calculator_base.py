__author__ = "Tomasz Rybotycki"

"""
    This class should be used as a base class for all standard BS permanent calculators.
    By standard I mean that the matrix and in(out)put states are stored in a variables.
    It takes care of a lot of boilerplate code.
"""

from typing import Optional, List

import operator
from functools import reduce

from numpy import ndarray, int64, array, asarray, zeros, ones, complex128, nonzero

from ..permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)
import abc


class BSPermanentCalculatorBase(BSPermanentCalculatorInterface, abc.ABC):
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
        self._input_state = asarray(input_state, dtype=int64)

    @property
    def output_state(self) -> ndarray:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self._output_state = asarray(output_state, dtype=int64)

    def _can_calculation_be_performed(self) -> bool:
        """
        Checks if calculation can be performed. For this to happen sizes of given
        matrix and states have to match.

        :return: Information if the calculation can be performed.
        """
        return (
            self._matrix.shape[0] == self._matrix.shape[1]
            and len(self._output_state) == len(self._input_state)
            and len(self._output_state) <= self._matrix.shape[0]
        )


class BSGuanCodeBasedPermanentCalculatorBase(BSPermanentCalculatorBase, abc.ABC):
    """
    This is a base class for those permanent calculators that use Guan codes iterations.
    It contains boilerplate code related to Guan codes iterations.
    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:
        super().__init__(matrix, input_state, output_state)

        # Guan codes-related variables
        self._r_vector: ndarray = zeros(len(self._input_state), dtype=int)  # g
        self._code_update_information: ndarray = ones(
            len(self._input_state), dtype=int
        )  # u
        self._position_limits: List[int] = list(self._input_state)  # n
        self._index_to_update: int = 0
        self._last_value_at_index: int = 0

        self._binomials_product: int = 1
        self.permanent: complex128

    def _initialize_permanent_computation(self) -> None:
        """Prepares the calculator for permanent computation."""

        self._multiplier = 1
        self._considered_columns_indices = nonzero(self._output_state)[0]
        self.permanent = complex128(0)

        self._sums = dict()

        self._binomials_product = 1
        self._initialize_guan_codes_variables()

    def _initialize_guan_codes_variables(self) -> None:
        """
        Initializes Guan codes-related variables before the permanents computation.
        """
        self._r_vector = zeros(len(self._input_state), dtype=int)  # g
        self._code_update_information = ones(len(self._input_state), dtype=int)  # u
        self._position_limits = list(self._input_state)  # n

    def _update_guan_code(self) -> None:
        """
        Prepare the class for processing the next Guan code.
        """
        self._index_to_update = 0  # i
        updated_value_at_index = (
            self._r_vector[0] + self._code_update_information[0]
        )  # k
        while (
            updated_value_at_index > self._position_limits[self._index_to_update]
            or updated_value_at_index < 0
        ):
            self._code_update_information[
                self._index_to_update
            ] = -self._code_update_information[self._index_to_update]
            self._index_to_update += 1

            if self._index_to_update == len(self._r_vector):
                return

            updated_value_at_index = (
                self._r_vector[self._index_to_update]
                + self._code_update_information[self._index_to_update]
            )

        self._last_value_at_index = self._r_vector[self._index_to_update]
        self._r_vector[self._index_to_update] = updated_value_at_index

    def _update_binomials_product(self) -> None:
        """
        Update the binomials product to reflect the new Guan code instead of recomputing
        it.
        """
        if self._r_vector[self._index_to_update] > self._last_value_at_index:
            self._binomials_product *= (
                self._input_state[self._index_to_update] - self._last_value_at_index
            ) / self._r_vector[self._index_to_update]
        else:
            self._binomials_product *= self._last_value_at_index / (
                self._input_state[self._index_to_update]
                - self._r_vector[self._index_to_update]
            )

    def compute_permanent(self) -> complex128:
        """
        This is the main method of the calculator. Assuming that input state,
        output state and the matrix are defined correctly (that is we've got m x m
        matrix, and vectors of with length m) this calculates the permanent of an
        effective scattering matrix related to probability of obtaining output state
        from a given input state.

        Most of the Guan code-related have essentially the same structure, up to some
        minor differences.

        :return: Permanent of effective scattering matrix.
        """
        if not self._can_calculation_be_performed():
            raise AttributeError

        self._initialize_permanent_computation()

        # Rest of the steps.
        while self._r_vector[-1] <= self._position_limits[-1]:

            self._update_guan_code()

            if self._index_to_update == len(self._r_vector):
                return self._return_permanent()

            # START PERMANENT UPDATE
            self._multiplier = -self._multiplier
            self._update_sums()
            self._update_binomials_product()
            self._update_permanent()

        return self._return_permanent()

    def _update_permanent(self) -> None:
        """Update permanent with new addend from r_vector."""
        self.permanent += reduce(
            operator.mul,
            [
                pow(self._sums[j], self._output_state[j])
                for j in self._considered_columns_indices
            ],
            self._multiplier * self._binomials_product,
        )

    @abc.abstractmethod
    def _update_sums(self) -> None:
        """An abstract method for sums update."""
        raise NotImplementedError

    @abc.abstractmethod
    def _return_permanent(self) -> complex128:
        """Some methods require additional operations"""
        raise NotImplementedError
