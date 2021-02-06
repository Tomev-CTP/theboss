__author__ = "Tomasz Rybotycki"

"""
    This file contains implementation of approximate boson sampling strategy subject to non-uniform losses. This can be
    well used to approximate boson sampling experiments with non-balanced network. More details can be found in [2].
"""

from typing import List

from numpy import ndarray, eye, exp, complex128, zeros
from numpy.random import rand

from src.quantum_computations_utilities import compute_qft_matrix

from src.simulation_strategies.lossy_networks_generalized_cliffords_simulation_strategy import \
    LossyNetworksGeneralizedCliffordsSimulationStrategy, BSPermanentCalculatorInterface

from src.boson_sampling_utilities.boson_sampling_utilities import \
    prepare_interferometer_matrix_in_expanded_space_with_first_k_lossless_modes


class NonuniformLossesApproximationStrategy(LossyNetworksGeneralizedCliffordsSimulationStrategy):

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface, approximated_modes_number: int) -> None:

        if approximated_modes_number > bs_permanent_calculator.matrix.shape[0]:
            approximated_modes_number = bs_permanent_calculator.matrix.shape[0]
        if approximated_modes_number < 0:
            approximated_modes_number = 0

        self._approximated_modes_number = approximated_modes_number
        self._initial_matrix = prepare_interferometer_matrix_in_expanded_space_with_first_k_lossless_modes\
            (bs_permanent_calculator.matrix, self._approximated_modes_number)
        super().__init__(bs_permanent_calculator)

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:

        samples = []

        for _ in range(samples_number):
            self._helper_strategy.set_new_matrix(self._get_matrix_for_approximate_sampling())
            samples.append(super().simulate(input_state)[0])

        return samples

    def _get_matrix_for_approximate_sampling(self) -> ndarray:
        qft_matrix = self._get_qft_matrix()
        random_phases_matrix = self._get_random_phases_matrix()
        return qft_matrix @ random_phases_matrix @ self._initial_matrix

    def _get_qft_matrix(self):
        small_qft_matrix = compute_qft_matrix(self._approximated_modes_number)
        qft_matrix = eye(self._initial_matrix.shape[0], dtype=complex128)

        for i in range(self._approximated_modes_number):
            for j in range(self._approximated_modes_number):
                qft_matrix[i][j] = small_qft_matrix[i][j]

        return qft_matrix

    def _get_random_phases_matrix(self) -> ndarray:
        random_phases_matrix = eye(self._initial_matrix.shape[0], dtype=complex128)

        for i in range(self._approximated_modes_number):
            random_phases_matrix[i][i] = exp(1j * rand())

        return random_phases_matrix
