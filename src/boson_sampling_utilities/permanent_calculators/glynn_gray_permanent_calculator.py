__author__ = "Tomasz Rybotycki"

"""
    This files contains the implementation of Glynn's formula for permanent calculation,
    that uses Gray codes to during the iterations. This method has the same complexity
    as the Ryser formula, but has been proven to be numerically more stable (due to
    physical limitations).
"""

from typing import Optional

from numpy import ndarray, complex128

from .bs_permanent_calculator_base import BSPermanentCalculatorBase


class GlynnGrayPermanentCalculator(BSPermanentCalculatorBase):

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        super().__init__(matrix, input_state, output_state)

    def compute_permanent(self) -> complex128:
        """
            This is the method for computing the permanent. It will work as described
            in [0], meaning implement formula (1) and iterate over deltas in Gray code.
        """
        permanent = complex128(0)



        return permanent

    def _generate_list_of_indices_to_update(self):
        pass