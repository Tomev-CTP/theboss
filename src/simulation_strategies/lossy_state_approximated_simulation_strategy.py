__author__ = "Tomasz Rybotycki"

"""
    The aim of this class is to produce an implementation of Boson Sampling algorithm
    as in [2], but for uniform losses. This would allow us to apply the losses to the
    k non-approximated particles and, by that, use generalized Cliffords strategy on
    the non-lossy interferometer. We won't need to expand the matrix into 2m x 2m!
"""

from .simulation_strategy_interface import SimulationStrategyInterface
from .generalized_cliffords_simulation_strategy import GeneralizedCliffordsSimulationStrategy, BSPermanentCalculatorInterface
from numpy import ndarray, hstack, zeros_like, complex128, eye, pi, ones, exp, diag
from numpy.random import choice, rand
from typing import Dict, List
from scipy.special import binom
from ..boson_sampling_utilities.boson_sampling_utilities import generate_lossy_inputs
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from ..quantum_computations_utilities import compute_qft_matrix

class LossyStateApproximationSimulationStrategy(SimulationStrategyInterface):
    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface, uniform_transmissivity: float,
                 approximated_modes_number: int, threads_number: int = -1):
        # Required for lossy approximated state preparation
        self._approximated_input_state_part_possibilities = None
        self._approximated_input_state_part_possibilities_weights = None
        self._not_approximated_lossy_mixed_state_parts = None
        self._not_approximated_lossy_mixed_state_parts_weights = None

        # Required for general simulation
        self._approximated_modes_number = approximated_modes_number
        self._uniform_transmissivity = uniform_transmissivity
        self._threads_number = self._get_proper_threads_number(threads_number)
        self._permanent_calculator = bs_permanent_calculator  # Should contain unitary (no losses here!)

    def _get_proper_threads_number(self, threads_number: int) -> int:
        if threads_number < 1 or threads_number > cpu_count():
            return cpu_count()
        else:
            return threads_number

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[
        ndarray]:

        if samples_number < 1:
            return []

        self._prepare_not_approximated_lossy_mixed_state(
            input_state[self._approximated_modes_number:]  # Not approximated state part
        )

        self._prepare_approximated_input_state(
            input_state[:self._approximated_modes_number]  # Approximated state part
        )

        number_of_samples_for_each_thread = \
            self._compute_number_of_samples_for_each_thread(samples_number)

        with Pool() as p:
            samples_lists = p.map(self._simulate_in_parallel,
                                  number_of_samples_for_each_thread)

        samples = [sample for samples_list in samples_lists for sample in samples_list]

        return samples

    def _prepare_not_approximated_lossy_mixed_state(
            self, not_approximated_input_state_part: ndarray) -> None:
        self._prepare_not_approximated_lossy_mixed_state_parts(not_approximated_input_state_part)
        self._prepare_not_approximated_lossy_mixed_state_parts_weights()

    def _prepare_not_approximated_lossy_mixed_state_parts(
            self, not_approximated_input_state_part: ndarray) -> None:
        self._not_approximated_lossy_mixed_state_parts = []
        for number_of_particles_left in range(sum(not_approximated_input_state_part) + 1):
            self._not_approximated_lossy_mixed_state_parts.extend(generate_lossy_inputs(
                not_approximated_input_state_part, number_of_particles_left))

    def _prepare_not_approximated_lossy_mixed_state_parts_weights(self) -> None:
        # Do note that this method HAS TO be called after lossy mixed state parts are
        # computed.
        possible_weights = self._get_possible_lossy_inputs_weights(
            # Last part is always whole (not approximated) input state part
            self._not_approximated_lossy_mixed_state_parts[-1]
        )
        self._not_approximated_lossy_mixed_state_parts_weights = []

        n = sum(self._not_approximated_lossy_mixed_state_parts[-1])

        for state_part in self._not_approximated_lossy_mixed_state_parts:
            self._not_approximated_lossy_mixed_state_parts_weights.append(
                possible_weights[int(sum(state_part))] / binom(n, sum(state_part))
            )

    def _get_possible_lossy_inputs_weights(
            self, input_state: ndarray) -> List[float]:
        weights = []

        # I'll use the same notation as in [1], for readability.
        n = int(sum(input_state))  # Initial number of particles.
        eta = self._uniform_transmissivity
        for l in range(n + 1):
            # l denotes number of particles left in the state
            weights.append(binom(n, l) * eta ** l * (1 - eta) ** (n - l))

        return weights

    def _prepare_approximated_input_state(
            self, approximated_input_state_part: ndarray) -> None:
        self._prepare_approximated_input_state_parts(approximated_input_state_part)
        self._prepare_approximated_input_state_parts_weights()

    def _prepare_approximated_input_state_parts(
            self, approximated_input_state_part: ndarray) -> None:
        self._approximated_input_state_part_possibilities = []
        for number_of_particles_left in range(int(sum(approximated_input_state_part)) + 1):
            state_part_possibility = zeros_like(approximated_input_state_part)
            state_part_possibility[-1] = number_of_particles_left
            self._approximated_input_state_part_possibilities.append(
                state_part_possibility
            )

    def _prepare_approximated_input_state_parts_weights(self):
        self._approximated_input_state_part_possibilities_weights = \
            self._get_possible_lossy_inputs_weights(
                # Last part contains all possible particles.
                self._approximated_input_state_part_possibilities[-1]
            )

    def _compute_number_of_samples_for_each_thread(self, samples_number: int) -> List[int]:
        # This method is basically the same as in Nonuniform losses approximation strategy
        samples_per_thread = (samples_number + self._threads_number - (
                    samples_number % self._threads_number))
        samples_per_thread = int(samples_per_thread / self._threads_number)
        number_of_samples_for_each_thread = [samples_per_thread] * self._threads_number

        return number_of_samples_for_each_thread

    def _simulate_in_parallel(self, samples_number: int = 1):
        samples = []

        helper_strategy = GeneralizedCliffordsSimulationStrategy(deepcopy(self._permanent_calculator))

        for _ in range(samples_number):
            lossy_input = self._get_input_state_for_sampling()
            approximate_sampling_matrix = self._get_matrix_for_approximate_sampling()

            helper_strategy.set_new_matrix(approximate_sampling_matrix)
            samples.append(helper_strategy.simulate(lossy_input)[0])

        return samples

    def _get_input_state_for_sampling(self):

        approximated_part = self._approximated_input_state_part_possibilities[choice(
            range(len(self._approximated_input_state_part_possibilities)),
            p=self._approximated_input_state_part_possibilities_weights
        )]

        print(self._not_approximated_lossy_mixed_state_parts)
        print(self._not_approximated_lossy_mixed_state_parts_weights)
        print(sum(self._not_approximated_lossy_mixed_state_parts_weights))

        for i in range(len(self._not_approximated_lossy_mixed_state_parts)):
            print(f""
                  f"{self._not_approximated_lossy_mixed_state_parts[i]}:"
                  f" {self._not_approximated_lossy_mixed_state_parts_weights[i]}")

        not_approximated_part = self._not_approximated_lossy_mixed_state_parts[choice(
            range(len(self._not_approximated_lossy_mixed_state_parts)),
            p=self._not_approximated_lossy_mixed_state_parts_weights
        )]
        return hstack([approximated_part, not_approximated_part])

    def _get_matrix_for_approximate_sampling(self) -> ndarray:
        # TODO TR: THIS WILL BE REWRITTEN AFTER MERGING WITH BRUTE-FORCE BRANCH
        qft_matrix = self._get_qft_matrix()
        random_phases_matrix = self._get_random_phases_matrix()
        return self._permanent_calculator.matrix @ random_phases_matrix @ qft_matrix

    def _get_qft_matrix(self):
        small_qft_matrix = compute_qft_matrix(self._approximated_modes_number)
        qft_matrix = eye(self._permanent_calculator.matrix.shape[0], dtype=complex128)

        qft_matrix[0:self._approximated_modes_number,
        0:self._approximated_modes_number] = small_qft_matrix

        return qft_matrix

    def _get_random_phases_matrix(self) -> ndarray:
        random_phases = ones(self._permanent_calculator.matrix.shape[0],
                             dtype=complex128)  # [1, 1,1 ,1,1,1]

        random_phases[0: self._approximated_modes_number] = exp(
            1j * 2 * pi * rand(self._approximated_modes_number))

        return diag(random_phases)
