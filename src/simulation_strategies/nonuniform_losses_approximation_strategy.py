__author__ = "Tomasz Rybotycki"

"""
    This file contains implementation of approximate boson sampling strategy subject to non-uniform losses. This can be
    well used to approximate boson sampling experiments with non-balanced network. More details can be found in [2].
"""

from copy import deepcopy
from random import uniform
from typing import List

from scipy import special

from numpy import complex128, exp, eye, ndarray, ones, diag, ones_like
from numpy.random import rand, choice

from .lossy_networks_generalized_cliffords_simulation_strategy import BSPermanentCalculatorInterface, \
    LossyNetworksGeneralizedCliffordsSimulationStrategy
from ..boson_sampling_utilities.boson_sampling_utilities import get_modes_transmissivity_values_from_matrix, \
    prepare_interferometer_matrix_in_expanded_space
from ..quantum_computations_utilities import compute_qft_matrix


class NonuniformLossesApproximationStrategy(LossyNetworksGeneralizedCliffordsSimulationStrategy):

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface, approximated_modes_number: int,
                 modes_transsmisivity: float) -> None:

        if approximated_modes_number > bs_permanent_calculator.matrix.shape[0]:
            approximated_modes_number = bs_permanent_calculator.matrix.shape[0]
        if approximated_modes_number < 0:
            approximated_modes_number = 0

        self._approximated_modes_number = approximated_modes_number
        self._modes_transmissivity = modes_transsmisivity
        self._initial_matrix = prepare_interferometer_matrix_in_expanded_space(bs_permanent_calculator.matrix)

        loss_removing_matrix = ones_like(self._initial_matrix[0])
        loss_removing_matrix[:approximated_modes_number] = 1.0 / self._modes_transmissivity
        loss_removing_matrix = diag(loss_removing_matrix)

        eta = self._modes_transmissivity
        k = approximated_modes_number

        self.binom_weights = [ ]
        weight = lambda l: pow(eta, l) * special.binom(k, l) * pow(1.0 - eta, k - l)
        for i in range(k + 1):
            self.binom_weights.append(weight(i))

        self._initial_matrix = self._initial_matrix @ loss_removing_matrix

        super().__init__(bs_permanent_calculator)

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:

        samples = []

        for _ in range(samples_number):
            lossy_input = self._compute_lossy_input(input_state)
            self._helper_strategy.set_new_matrix(self._get_matrix_for_approximate_sampling())
            samples.append(super().simulate(lossy_input)[0])

        return samples

    def _compute_lossy_input(self, input_state: ndarray) -> ndarray:

        if self._approximated_modes_number < 1:
            return input_state

        lossy_input = deepcopy(input_state)

        binned_input_index = self._approximated_modes_number - 1
        lossy_input[binned_input_index] = choice(range(self._approximated_modes_number + 1), p=self.binom_weights)

        return lossy_input

    def _get_matrix_for_approximate_sampling(self) -> ndarray:
        qft_matrix = self._get_qft_matrix()
        random_phases_matrix = self._get_random_phases_matrix()
        return self._initial_matrix @ random_phases_matrix @ qft_matrix

    def _get_qft_matrix(self):
        small_qft_matrix = compute_qft_matrix(self._approximated_modes_number)
        qft_matrix = eye(self._initial_matrix.shape[0], dtype=complex128)

        qft_matrix[0:self._approximated_modes_number, 0:self._approximated_modes_number] = small_qft_matrix

        return qft_matrix

    def _get_random_phases_matrix(self) -> ndarray:
        random_phases = ones(self._initial_matrix.shape[0], dtype=complex128)

        random_phases[0: self._approximated_modes_number] = exp(1j * rand(self._approximated_modes_number))

        return diag(random_phases)
