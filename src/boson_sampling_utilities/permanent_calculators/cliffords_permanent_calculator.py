from bs_permanent_calculator_base import BSPermanentCalculatorBase
from numpy import ndarray, complex128, nonzero
from typing import Optional, List

# TR TODO: This method is not yet finished as I were tasked with different things.


class CliffordsPermanentCalculator(BSPermanentCalculatorBase):
    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        super().__init__(matrix, input_state, output_state)
        self._submatrices_permanents = []
        self._forward_cumulative_products = []
        self._backward_cumulative_products = []

    def compute_permanent(self) -> complex128:

        if not self._can_calculation_be_performed():
            raise AttributeError

        if not self._submatrices_permanents:
            self._compute_submatrices_permanents()
        else:
            # This should mean that new output varies from new in only one place,
            # meaning that new matrix varies in only one column. According to Cliffords
            # this allows us to somehow modify the permanents of submatrices in O(k),
            # giving us the result faster. I do not know how to modify them YET.
            pass

        permanent = 0
        # Write the sum here. Would it be as trivial as in Cliffords work?
        return permanent

    def _compute_cumulative_products(self) -> None:
        self._compute_forward_cumulative_products()
        self._compute_backward_cumulative_products()

    def _compute_forward_cumulative_products(self) -> None:
        # I'll be using same name convention as in Cliffords2020
        self._forward_cumulative_products = [1]




    def _compute_backward_cumulative_products(self) -> None:
        pass

    def _compute_submatrices_permanents(self) -> None:
        # We first want to get the reduced matrix. This means we want to remove the last
        # column from the effective scattering matrix (B in Cliffords). This means we
        # need to remove last particle from the output state.
        new_output = self._get_new_output()
        # We now want to consider submatrices with created by taking every column from
        # the effective scattering matrix. This means that we want to take every one
        # particle from every occupied mode in the input state.
        new_inputs = self._get_new_inputs()

        # Now I basically proceed with the same approach as in Ryser-Guan. There will,
        # however, be difference in how I will compute (or rather update) the sums
        # and the products, as I will be using forward/backward cumulative products
        # as in Cliffords2020.
        for input_state in new_inputs:
            pass

    def _compute_submatrix_permanent(self):
        pass

    def _get_new_output(self) -> ndarray:
        new_output = self._output_state
        last_nonzero_index = max(nonzero(self._output_state)[0])
        new_output[last_nonzero_index] -= 1
        return new_output

    def _get_new_inputs(self) -> List[ndarray]:
        inputs_non_zero_indices = nonzero(self._input_state)[0]
        new_inputs = []
        for non_zero_index in inputs_non_zero_indices:
            new_input = self._input_state
            new_input[non_zero_index] -= 1
            new_inputs.append(new_input)
        return new_inputs
