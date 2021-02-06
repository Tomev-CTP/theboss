__author__ = "Tomasz Rybotycki"

"""
    This class is meant to parallelize the CH Permanent Calculator.
"""

from multiprocessing.pool import Pool
from typing import Optional

from numpy import complex128, ndarray

from src.boson_sampling_utilities.permanent_calculators.chin_huh_permanent_calculator import ChinHuhPermanentCalculator


class ParallelChinHuhPermanentCalculator(ChinHuhPermanentCalculator):

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        super().__init__(matrix, input_state, output_state)

    def calculate(self) -> complex128:
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.

            This method uses multiprocessing!

            :return: Permanent of effective scattering matrix.
        """
        if not self._can_calculation_be_performed():
            raise AttributeError

        v_vectors = self._calculate_v_vectors()

        permanent = complex128(0)

        with Pool() as p:
            results = p.map(self._compute_permanent_addend, v_vectors)
        permanent += sum(results)
        permanent /= pow(2, sum(self._input_state))

        return permanent
