__author__ = "Tomasz Rybotycki"

"""
    This script contains base(s) for BS submatrices permanents calculators that takes
    care of the boilerplate code.
"""


from theboss.boson_sampling_utilities.permanent_calculators.bs_submatrices_permanent_calculator_interface import (
    BSSubmatricesPermanentCalculatorInterface,
)
from typing import Optional, List
from numpy import ndarray, int64, array, zeros, ones
import abc


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


class BSGuanBasedSubmatricesPermanentCalculatorBase(
    BSSubmatricesPermanentCalculatorBase, abc.ABC
):
    """
    This is a base class for BS submatrices permanents calculators that use Guan code
    iterations for computations. Submatrices permanents computations are usually some
    variations of C&C idea, so it's pretty common.
    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:

        super().__init__(matrix, input_state, output_state)

        # Guan codes related
        self._index_to_update: int = 0
        self._last_value_at_index: int = 0

        self._position_limits: List[int]
        self._r_vector: ndarray
        self._code_update_information: ndarray

        self.initialize_guan_codes_variables()

        self._binomials_product: int = 1

    def initialize_guan_codes_variables(self) -> None:
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