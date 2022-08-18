__author__ = "Tomasz Rybotycki"

"""
    This file contains the implementation of mean field boson sampling strategy subject
    to non-uniform losses. This can be used to approximate boson sampling experiments
    with non-balanced networks. More details can be found in [2].
"""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy
from math import sqrt
from multiprocessing import cpu_count
from typing import List, Sequence, Dict, Tuple

from numpy import ndarray, diag, isclose
from numpy.random import randint
from theboss.math_utilities import choice
from numpy.linalg import svd
from scipy.special import binom

from .generalized_cliffords_nonuniform_losses_simulation_strategy import (
    BSPermanentCalculatorInterface,
    GeneralizedCliffordsNonuniformLossesSimulationStrategy,
)
from theboss.boson_sampling_utilities import (
    generate_qft_matrix_for_first_m_modes,
    generate_random_phases_matrix_for_first_m_modes,
)
from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)


class GeneralizedMeanFieldNonuniformLossesSimulationStrategy(
    SimulationStrategyInterface
):
    """
    This is an implementation of the algorithm presented by Brod and Oszmaniec in their
    2020 work [2]. Without the loss of generalization we assume that the first :math:`k`
    modes will be approximated.
    """

    def __init__(
        self,
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        approximated_modes_number: int,
        threads_number: int = -1,
    ) -> None:

        self._approximated_modes_number = self._get_proper_approximated_modes_number(
            bs_permanent_calculator, approximated_modes_number
        )

        self._uniform_losses: float = 0
        self._initial_matrix: Sequence[Sequence[complex]] = list(list())

        self._extract_losses_from_the_interferometer(bs_permanent_calculator.matrix)

        # Fill weights at the beginning of the simulation, when input is given.
        self._binomial_weights: Dict[int, List[float]] = dict()
        self._threads_number = self._get_proper_threads_number(threads_number)
        self._permanent_calculator = bs_permanent_calculator

        self._state_without_approximated_modes: List[int] = list()
        self._approximated_modes_particles_number: int = 0

    @staticmethod
    def _get_proper_approximated_modes_number(
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        approximated_modes_number: int,
    ):
        """
        Bounds the approximated modes number to the proper values.

        :param bs_permanent_calculator:
            Permanent calculator. Required do assess the total number of modes.
        :param approximated_modes_number:
            Number of approximated modes specified by the user.

        :return:
            Properly bounded number of approximated modes.
        """
        total_number_of_modes: int = len(bs_permanent_calculator.matrix)

        if approximated_modes_number > total_number_of_modes:
            approximated_modes_number = total_number_of_modes
        if approximated_modes_number < 0:
            approximated_modes_number = 0
        return approximated_modes_number

    def _extract_losses_from_the_interferometer(
        self, interferometer_matrix: Sequence[Sequence[complex]]
    ) -> None:
        """
        Extracts maximal amount of uniform losses from the interferometer_matrix.

        :param interferometer_matrix:
            Possibly lossy (sub)unitary interferometer matrix.
        """
        u: ndarray
        s: List[float]
        v: ndarray
        u, s, v = svd(interferometer_matrix)

        # Extract uniform losses from the matrix
        transmissivities: List[float] = [singular_value**2 for singular_value in s]
        losses: List[float] = [1 - eta for eta in transmissivities]
        self._uniform_losses = min(losses)

        # Form the interferometer_matrx with losses extracted.
        transmissivities = [
            eta / (1 - self._uniform_losses) for eta in transmissivities
        ]
        s = [sqrt(eta) for eta in transmissivities]
        self._initial_matrix = u @ diag(s) @ v

    def _compute_binomial_weights(
        self, max_particles_number: int
    ) -> Dict[int, List[float]]:
        """
        Prepares a dict of list of binomial weights for sampling proper number of
        particles in a mode after application of uniform losses extracted from the
        interferometer matrix.

        :param max_particles_number:
            Maximal number of particles in a single mode of the input state. Recall that
            we expect the input state to already be in the proper form, e.g. with the
            bunching prepared according to the algorithm specification [2].

        :return:
            A dict of lists of binomial weights that one can use to sample the number
            of particles left after application of the uniform losses.
        """
        eta = 1 - self._uniform_losses  # Uniform transmissivity

        binomial_weights: Dict[int, List[float]] = {}

        # Just a shorthand notation.
        def binomial_weight(total_particles: int, particles_left: int) -> float:
            """

            :param total_particles:
                Total number of particles.
            :param particles_left:
                Number of particles left.
            :return:
                Probability of getting :math:`l` particles after application of uniform
                losses to :math:`n` particles given transmissivity :math:`\\eta`.
            """
            return (
                pow(eta, particles_left)
                * binom(total_particles, particles_left)
                * pow(1.0 - eta, total_particles - particles_left)
            )

        for n in range(1, max_particles_number + 1):
            binomial_weights[n] = []
            for particles_left in range(n + 1):
                binomial_weights[n].append(binomial_weight(n, particles_left))

        return binomial_weights

    @staticmethod
    def _get_proper_threads_number(threads_number: int) -> int:
        """
        For multithreading. Shorthand notation for is that for the number exceeding
        the total cpu_count() or the specified threads number is negative, then maximal
        possible threads number (cpu_count()) is computed.

        :param threads_number:
             Threads number given by the user.

        :return:
            Possibly fixed number of threads.
        """
        if threads_number < 1 or threads_number > cpu_count():
            return cpu_count()
        else:
            return threads_number

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Sequence[int]]:
        """
        Main method of the simulator. It samples from the generalized mean-field BS
        distribution that is specified by the input_state and the previously given
        interferometer matrix.

        :param input_state:
            Fock state in the 2nd quantization description.
        :param samples_number:
            The number of samples that will be returned.

        :return:
            Specified number of samples from the generalized mean-field BS distribution.
        """
        if samples_number < 1:
            return list()

        # Prepare the state used in the approximate simulation.
        self._state_without_approximated_modes = list(input_state)

        for i in range(self._approximated_modes_number):
            self._approximated_modes_particles_number += input_state[i]
            self._state_without_approximated_modes[i] = 0

        maximum_particles_in_mode: int = max(
            max(input_state), self._approximated_modes_particles_number
        )
        self._binomial_weights = self._compute_binomial_weights(
            maximum_particles_in_mode
        )

        # Get samples number per thread
        samples_per_thread = (
            samples_number
            + self._threads_number
            - (samples_number % self._threads_number)
        )
        samples_per_thread = int(samples_per_thread / self._threads_number)
        samples_for_threads = [samples_per_thread] * self._threads_number

        # Context is required on Linux systems, as the default (fork) produces undesired
        # results! Spawn is default on osX and Windows and works as expected.
        multiprocessing_context = multiprocessing.get_context("spawn")

        with Pool(mp_context=multiprocessing_context) as p:
            samples_lists = p.map(self._simulate_in_parallel, samples_for_threads)

        samples = [sample for samples_list in samples_lists for sample in samples_list]

        return samples

    def _simulate_in_parallel(self, samples_number: int = 1) -> List[Sequence[int]]:
        """
        A part of simulation that can be performed independently in separate threads. It
        creates samples using GCCB strategy destined for lossy networks.

        :param samples_number:
            The number of samples that a single thread should sample.

        :return:
            A list of sampled output states.
        """
        samples: List[Tuple[int, ...]] = []

        helper_strategy: GeneralizedCliffordsNonuniformLossesSimulationStrategy = (
            GeneralizedCliffordsNonuniformLossesSimulationStrategy(
                deepcopy(self._permanent_calculator)
            )
        )

        for _ in range(samples_number):
            approximate_state = deepcopy(self._state_without_approximated_modes)

            # Symmetrization.
            if self._approximated_modes_number > 0:
                approximate_state[
                    randint(0, self._approximated_modes_number)
                ] = self._approximated_modes_particles_number

            lossy_approximate_input_state = self._compute_lossy_input(approximate_state)

            approximate_sampling_matrix = self._get_matrix_for_approximate_sampling()

            helper_strategy.set_new_matrix(approximate_sampling_matrix)
            samples.append(helper_strategy.simulate(lossy_approximate_input_state)[0])

        return samples

    def _compute_lossy_input(self, input_state: Sequence[int]) -> Tuple[int, ...]:
        """
        Applies the initial channel of extracted uniform losses to the input state.

        :param input_state:
            Input state to which uniform losses channel will be applied.

        :return:
            Lossy input state.
        """

        # If there are no uniform losses at the beginning, then the input cannot be
        # lossy. All potential losses are in the network.
        if isclose(self._uniform_losses, 0):
            return tuple(input_state)

        lossy_input = tuple()

        for mode in range(len(input_state)):

            if input_state[mode] == 0:
                lossy_input += (0,)
                continue

            lossy_input += (
                choice(
                    list(range(input_state[mode] + 1)),
                    self._binomial_weights[input_state[mode]],
                ),
            )

        return lossy_input

    def _get_matrix_for_approximate_sampling(self) -> ndarray:
        """
        Generates the matrix for the approximate simulation. Do note that this matrix
        has to be computed for each sample, as we have to apply random phases for
        each sample.

        :return:
            Returns an array for the matrix
        """
        qft_matrix = generate_qft_matrix_for_first_m_modes(
            self._approximated_modes_number, len(self._initial_matrix)
        )
        random_phases_matrix = generate_random_phases_matrix_for_first_m_modes(
            self._approximated_modes_number, len(self._initial_matrix)
        )

        return self._initial_matrix @ random_phases_matrix @ qft_matrix
