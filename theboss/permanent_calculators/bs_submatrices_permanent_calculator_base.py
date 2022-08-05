__author__ = "Tomasz Rybotycki"

"""
    This script contains base(s) for BS submatrices permanents calculators that takes
    care of the boilerplate code.
"""


from theboss.permanent_calculators.bs_submatrices_permanent_calculator_interface import (
    BSSubmatricesPermanentCalculatorInterface,
)
from typing import Optional, List, Sequence
from numpy import ndarray, zeros, ones, complex128, nonzero
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
        matrix: Sequence[Sequence[complex]],
        input_state: Optional[Sequence[int]] = None,
        output_state: Optional[Sequence[int]] = None,
    ) -> None:
        if output_state is None:
            output_state = []
        if input_state is None:
            input_state = []
        self._matrix: Sequence[Sequence[complex]] = matrix
        self._input_state: Sequence[int] = input_state
        self._output_state: Sequence[int] = output_state

    @property
    def matrix(self) -> Sequence[Sequence[complex]]:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: Sequence[Sequence[complex]]) -> None:
        self._matrix = matrix

    @property
    def input_state(self) -> Sequence[int]:
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: Sequence[int]) -> None:
        self._input_state = input_state

    @property
    def output_state(self) -> Sequence[int]:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: Sequence[int]) -> None:
        self._output_state = output_state


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
        matrix: Sequence[Sequence[complex]],
        input_state: Optional[Sequence[int]] = None,
        output_state: Optional[Sequence[int]] = None,
    ) -> None:

        super().__init__(matrix, input_state, output_state)

        # Guan codes related
        self._multiplier = None
        self._index_to_update: int = 0
        self._last_value_at_index: int = 0

        self._position_limits: List[int]
        self._r_vector: ndarray
        self._code_update_information: ndarray

        self._binomials_product: int = 1

        self.permanent: complex128

    def _initialize_guan_codes_variables(self) -> None:
        """
        Initializes Guan codes-related variables before the permanent computation.
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

    def compute_permanents(self) -> List[complex128]:
        """
        The main method of the class. Computes the permanents of the submatrices by
        using Ryser's formula, the input-output exchange trick and the Guan codes.
        """

        # Take care of the edge-case, where only 1 sub-matrix is valid (and empty).
        if sum(self.input_state) == 1:
            return [complex128(v) for v in self.input_state]

        self._initialize_permanents_computation()

        while self._r_vector[-1] <= self._position_limits[-1]:

            self._update_guan_code()

            if self._index_to_update == len(self._r_vector):
                return self.permanents

            self._multiplier = -self._multiplier
            self._update_binomials_product()
            self._update_sums()

            self._update_permanents()

        return self.permanents

    def _initialize_permanents_computation(self) -> None:
        """
        A method initializing all the class fields. Should be called prior to
        the permanents computation.
        """
        self.permanents = [complex128(0) for _ in range(len(self.input_state))]

        self._sums = dict()
        self._multiplier = pow(-1, sum(self.output_state))
        self._considered_columns_indices = nonzero(self._output_state)[0]
        self._binomials_product = 1

        self._initialize_guan_codes_variables()

    @abc.abstractmethod
    def _update_sums(self) -> None:
        """
        Updates the sums instead of recomputing them.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _update_permanents(self) -> None:
        """Update permanents with new r_vector data. Notice that some of"""
        raise NotImplementedError
