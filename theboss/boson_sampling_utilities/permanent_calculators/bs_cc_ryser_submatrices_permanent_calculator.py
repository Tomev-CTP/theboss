__author__ = "Tomasz Rybotycki"

"""
    This file contains the implementation of collective submatrices permanents
    calculator using Ryser's formula and Guan codes iteration.
"""

from theboss.boson_sampling_utilities.permanent_calculators.bs_submatrices_permanent_calculator_base import (
    BSGuanBasedSubmatricesPermanentCalculatorBase,
)

from numpy import ndarray, complex128, nonzero
from typing import Optional, List


class BSCCRyserSubmatricesPermanentCalculator(
    BSGuanBasedSubmatricesPermanentCalculatorBase
):

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

    This way we reduce the computations in case of binned outputs. Also notice that
    we can just switch the indices instead of transposing the matrix. The resultant
    formula used in this implementation is

    .. math:: Perm(U_{ST}) = Perm(U^T_{TS}) =
        (-1)^k \\sum^{s_1}_{r_1} ... \\sum^{s_m}_{r_m} (-1)^{r_1 + ... r_m}
        \\prod^m_{\\nu = 1} \\binom{s_\\nu}{r_\\nu}
        \\prod_{j: t_j > 0} \\left (
            \\sum^m_{\\nu = 1} r_\\nu u{j, \\nu}
        \\right )^{t_j}.

    """

    def __init__(
        self,
        matrix: ndarray,
        input_state: Optional[ndarray] = None,
        output_state: Optional[ndarray] = None,
    ) -> None:

        super().__init__(matrix, input_state, output_state)

        self._sums: dict = dict()  # w_j(r)
        self._prod: complex128

        self._considered_columns_indices = nonzero(self._output_state)[0]
        self._multiplier = 1
        self._binomials_product = 1

        self.permanents: List[complex128] = []

    def _initialize_permanents_computation(self) -> None:
        """Prepare the calculator for the computations."""
        super()._initialize_permanents_computation()

        for j in self._considered_columns_indices:
            self._sums[j] = 0

    def _update_sums(self) -> None:
        """
        Updates the sums instead of recomputing them. Notice, moreover, that the
        product of the sums is also updated in the same loop.
        """
        self._prod = complex128(1)

        for j in self._sums:
            self._sums[j] += (
                self._r_vector[self._index_to_update] - self._last_value_at_index
            ) * self._matrix[j][self._index_to_update]

            self._prod *= pow(self._sums[j], self.output_state[j])

        self._prod *= self._multiplier

    def _update_permanents(self) -> None:
        """
        Add addends to the proper permanents. The i-th permanent (so the one missing i
        column) should be updated if r_vector[i] != input_vector[i] and when
        input_vector[i] != 0. Do note that it also requires updating the binomial
        product to
        """
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

            self.permanents[i] += self._prod * updated_binomials_product
