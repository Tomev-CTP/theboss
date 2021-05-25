__author__ = "Tomasz Rybotycki"

"""
    This file contains implementation of approximate boson sampling strategy subject to non-uniform losses. This can be
    well used to approximate boson sampling experiments with non-balanced network. More details can be found in [2].
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from itertools import repeat
from math import sqrt
from multiprocessing import cpu_count
from typing import List

from numpy import ndarray, diag, ones_like
from numpy.random import choice
from scipy import special

from .lossy_networks_generalized_cliffords_simulation_strategy import \
    BSPermanentCalculatorInterface, \
    LossyNetworksGeneralizedCliffordsSimulationStrategy
from ..boson_sampling_utilities.boson_sampling_utilities import \
    prepare_interferometer_matrix_in_expanded_space, \
    generate_qft_matrix_for_first_m_modes, \
    generate_random_phases_matrix_for_first_m_modes


class NonuniformLossesApproximationStrategy():

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface,
                 approximated_modes_number: int,
                 modes_transsmisivity: float, threads_number: int = -1) -> None:

        self._approximated_modes_number = self._get_proper_approximated_modes_number(
            bs_permanent_calculator,
            approximated_modes_number)
        self._modes_transmissivity = modes_transsmisivity

        self._initial_matrix = self._prepare_initial_matrix(bs_permanent_calculator)

        self._binom_weights = self._compute_binomial_weights()

        self._threads_number = self._get_proper_threads_number(threads_number)

        self._permanent_calculator = bs_permanent_calculator

    @staticmethod
    def _get_proper_approximated_modes_number(
            bs_permanent_calculator: BSPermanentCalculatorInterface,
            approximated_modes_number: int):
        if approximated_modes_number > bs_permanent_calculator.matrix.shape[0]:
            approximated_modes_number = bs_permanent_calculator.matrix.shape[0]
        if approximated_modes_number < 0:
            approximated_modes_number = 0
        return approximated_modes_number

    def _prepare_initial_matrix(self,
                                bs_permanent_calculator: BSPermanentCalculatorInterface):

        loss_removing_matrix = ones_like(bs_permanent_calculator.matrix[0])
        loss_removing_matrix[:self._approximated_modes_number] = 1.0 / sqrt(
            self._modes_transmissivity) # This here assumes uniform losses
        loss_removing_matrix = diag(loss_removing_matrix)

        initial_matrix = bs_permanent_calculator.matrix @ loss_removing_matrix

        initial_matrix = prepare_interferometer_matrix_in_expanded_space(initial_matrix)

        return initial_matrix

    def _compute_binomial_weights(self):

        eta = self._modes_transmissivity
        k = self._approximated_modes_number

        binom_weights = []

        weight = lambda l: pow(eta, l) * special.binom(k, l) * pow(1.0 - eta, k - l)
        for i in range(k + 1):
            binom_weights.append(weight(i))

        return binom_weights

    def _get_proper_threads_number(self, threads_number: int) -> int:
        if threads_number < 1 or threads_number > cpu_count():
            return cpu_count()
        else:
            return threads_number

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:

        if samples_number < 1:
            return []

        # Get samples number per thread
        samples_per_thread = (samples_number + self._threads_number - (
                    samples_number % self._threads_number))
        samples_per_thread = int(samples_per_thread / self._threads_number)
        samples_for_threads = [samples_per_thread] * self._threads_number

        # Context is required on Linux systems, as the default (fork) produces undesired results! Spawn is default
        # on osX and Windows and works as expected.
        multiprocessing_context = multiprocessing.get_context("spawn")

        with Pool(mp_context=multiprocessing_context) as p:
            samples_lists = p.map(self._simulate_in_pararell, repeat(input_state),
                                  samples_for_threads)

        samples = [sample for samples_list in samples_lists for sample in samples_list]

        return samples

    def _simulate_in_parallel(self, input_state: ndarray, samples_number: int = 1):
        samples = []

        helper_strategy = LossyNetworksGeneralizedCliffordsSimulationStrategy(
            deepcopy(self._permanent_calculator))

        for _ in range(samples_number):
            lossy_input = self._compute_lossy_input(input_state)

            # if not array_equal(lossy_input, input_state):
            #    print(f"Got {lossy_input.__str__()}, expected: {input_state.__str__()}") # For k = # modes

            approximate_sampling_matrix = self._get_matrix_for_approximate_sampling()

            # if not array_equal(approximate_sampling_matrix, self._initial_matrix):
            # print(f"Got {approximate_sampling_matrix.__str__()}, expected: {self._initial_matrix.__str__()}")  # For k = # modes

            helper_strategy.set_new_matrix(approximate_sampling_matrix)
            samples.append(helper_strategy.simulate(lossy_input)[0])

        return samples

    def _compute_lossy_input(self, input_state: ndarray) -> ndarray:

        if self._approximated_modes_number < 1:
            return input_state

        lossy_input = deepcopy(input_state)

        binned_input_index = self._approximated_modes_number - 1
        lossy_input[binned_input_index] = choice(
            range(self._approximated_modes_number + 1), p=self._binom_weights)

        return lossy_input

    def _get_matrix_for_approximate_sampling(self) -> ndarray:
        qft_matrix = generate_qft_matrix_for_first_m_modes(
            self._approximated_modes_number,
            self._initial_matrix.shape[0])
        random_phases_matrix = generate_random_phases_matrix_for_first_m_modes(
            self._approximated_modes_number,
            self._initial_matrix.shape[0]
        )

        return self._initial_matrix @ random_phases_matrix @ qft_matrix
