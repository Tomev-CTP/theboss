__author__ = "Tomasz Rybotycki"

"""
    This file contains the implementation of collective submatrices permanents
    calculator 
"""

from theboss.boson_sampling_utilities.permanent_calculators.bs_cc_ch_submatrices_permanent_calculator import (
    BSSubmatricesPermanentCalculatorBase,
)

from numpy import ndarray, complex128, nonzero, zeros, ones
from typing import Optional, List, Union


class BSCCRyserSubmatricesPermanentCalculator(BSSubmatricesPermanentCalculatorBase):

    """
    This class computes the permanents of the submatrices using slightly modified
    Clifford & Clifford approach from the [5]. Here we utilize the fact that

    .. math:: Perm(U_{ST}) = Perm(U^T_{TS})

    so we can reverse the role of input and output during the permanent calculation.
    This approach is not desired for the standard BS experiment (wherein the input
    is not binned) but is definitely required for the approximate method [2]. For
    the purposes of m~=n experiments we will also modify the Ryser formula to take
    into account binning in the outputs. Let U be the interferometer matrix and let
    B be :math:`U^T_{TS}`, where :math:`|S>` and :math:`|T>` are the input and
    output states and T denotes the transposition. Moreover let
    :math:`u^T_{i, j} = [U^T]_{ij}`. Then

    .. math:: Perm(U_{ST}) = Perm(U^T_{TS}) =
        (-1)^k \\sum^{s_1}_{r_1} ... \\sum^{s_m}_{r_m} (-1)^{r_1 + ... r_m}
        \\prod^m_{\\nu = 1} \\binom{s_\\nu}{r_\\nu}
        \\prod_{j: t_j > 0} \\left (
            \\sum^m_{\\nu = 1} r_\\nu u^T_{\\nu, j}
        \\right )^{t_j}

    This way we reduce the computations in case of binned outputs.
    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:

        super().__init__(matrix, input_state, output_state)

        # We store the transposed matrix to reduce the computation time.
        self._matrix_t = matrix.transpose()

        self._sums: dict = dict()  # w_j(r)
        self._forward_products: Union[dict, None] = None
        self._backward_products: Union[dict, None] = None
        self._considered_columns_indices = nonzero(self._output_state)[0]
        self._multiplier = 1
        self._binomials_product = 1

        # Required for Guan Code iteration. Some values are also used during updates.
        self._r_vector: ndarray = zeros(len(self._input_state), dtype=int)  # g
        self._code_update_information: ndarray = ones(
            len(self._input_state), dtype=int
        )  # u
        self._position_limits: List[int] = list(self._input_state)  # n
        self._index_to_update: int = 0
        self._last_value_at_index: int = 0

        self.permanents: List[complex128] = []

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: ndarray) -> None:
        self._matrix = matrix
        # We store the transposed matrix to speed up the computations.
        self._matrix_t = matrix.transpose()

    def compute_permanents(self) -> List[complex128]:
        """
        The main method of the class. Computes the permanents of the submatrices by
        using Ryser's formula, the input-output exchange trick and the Guan codes.
        """
        self._initialize_permanents_computation()

        while self._r_vector[-1] <= self._position_limits[-1]:

            self._update_guan_code()

            if self._index_to_update == len(self._r_vector):
                return self.permanents

            self._update_permanents()

        return self.permanents

    def _initialize_permanents_computation(self) -> None:
        """
        A method initializing all the class fields. Should be called prior to
        the permanents computation.
        """
        self.permanents = [complex128(0) for _ in range(len(self.input_state))]

        self._sums = dict()
        self._forward_products = None
        self._backward_products = None
        self._multiplier = pow(-1, sum(self.output_state))
        self._considered_columns_indices = nonzero(self._output_state)[0]
        self._binomials_product = 1

        # Required for Guan Code iteration
        self._r_vector = zeros(len(self._input_state), dtype=int)  # g
        self._code_update_information = ones(len(self._input_state), dtype=int)  # u
        self._position_limits = list(self._input_state)  # n

        for j in self._considered_columns_indices:
            self._sums[j] = 0

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

    def _update_permanents(self) -> None:
        """
        Add addends to the proper permanents. The i-th permanent (so the one missing i
        column) should be updated if r_vector[i] != input_vector[i] and when
        input_vector[i] != 0. Do note that it also requires updating the binomial
        product to
        """
        self._multiplier = -self._multiplier

        prod: complex128 = complex128(1)

        # Sums update.
        for j in self._sums:
            self._sums[j] += (
                self._r_vector[self._index_to_update] - self._last_value_at_index
            ) * self._matrix_t[self._index_to_update][j]

            prod *= pow(self._sums[j], self.output_state[j])

        self._update_binomials_product()

        prod *= self._multiplier

        for i in range(len(self.permanents)):

            # We don't want to update permanents with the wrong r-vector sums.
            if 0 < self._r_vector[i] == self.input_state[i]:
                continue

            # We've got to update the binomials product to ensure that it reflects
            # the proper input state (which is the stored input state reduced by
            # the particle in the i-th mode).
            updated_binomials_product: float = self._binomials_product

            if self.input_state[i] != 0:
                updated_binomials_product /= self.input_state[i] / (
                    self.input_state[i] - self._r_vector[i]
                )

            self.permanents[i] += prod * updated_binomials_product

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
