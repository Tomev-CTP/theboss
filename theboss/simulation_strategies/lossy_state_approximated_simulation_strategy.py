__author__ = "Tomasz Rybotycki"

"""
    The aim of this class is to produce an implementation of Boson Sampling algorithm
    as in [2], but for uniform losses. This would allow us to apply the losses to the
    k non-approximated particles and, by that, use generalized Cliffords strategy on
    the non-lossy interferometer. We won't need to expand the matrix into 2m x 2m!
"""

from .simulation_strategy_interface import SimulationStrategyInterface
from .generalized_cliffords_b_simulation_strategy import (
    GeneralizedCliffordsBSimulationStrategy,
    BSPermanentCalculatorInterface,
)
from numpy import (
    ndarray,
    hstack,
    zeros_like,
    array,
    arange,
)
from numpy.random import choice, shuffle
from typing import List, Tuple, Sequence
from scipy.special import binom
from theboss.boson_sampling_utilities import (
    generate_lossy_n_particle_input_states,
    generate_qft_matrix_for_first_m_modes,
    generate_random_phases_matrix_for_first_m_modes,
)
from multiprocessing import cpu_count
import multiprocessing
from concurrent.futures import ProcessPoolExecutor as Pool
from copy import deepcopy


class LossyStateApproximationSimulationStrategy(SimulationStrategyInterface):
    """
    An implementation of the BOBS strategy [2] designed for the uniformly lossy
    experiments. It applies the losses to the state before the sampling begins.
    """

    def __init__(
        self,
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        uniform_transmissivity: float,
        hierarchy_level: int,
        threads_number: int = -1,
    ):
        # Required for lossy approximated state preparation
        self._approximated_input_state_part_possibilities = None
        self._approximated_input_state_part_possibilities_weights = None
        self._not_approximated_lossy_mixed_state_parts = None
        self._not_approximated_lossy_mixed_state_parts_weights = None

        # Required for general simulation
        self._hierarchy_level: int = hierarchy_level
        self._uniform_transmissivity: float = uniform_transmissivity
        self._threads_number: int = self._get_proper_threads_number(threads_number)
        self._permanent_calculator: BSPermanentCalculatorInterface = (
            bs_permanent_calculator  # Should contain a UNITARY (no losses here!)
        )
        self._qft_matrix = generate_qft_matrix_for_first_m_modes(
            len(bs_permanent_calculator.matrix) - hierarchy_level,
            len(bs_permanent_calculator.matrix),
        )

    @staticmethod
    def _get_proper_threads_number(threads_number: int) -> int:
        """
        Computes the proper number of thread, if the one specified by the user
        is a nonsense.

        Note: Maximal number of threads is given if the number specified by the user
        is negative.

        TODO TR: Consider putting it into some general file.

        :param threads_number:
            Threads number specified by the user.

        :return:
            The number of threads that the sampler will use.
        """
        if threads_number < 1 or threads_number > cpu_count():
            return cpu_count()
        else:
            return threads_number

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Generates a list of samples from the BS experiment instance.

        :param input_state:
            The input state of the BS experiment.

        :param samples_number:
            The number of samples that will be returned.

        :return:
            A list of sampled outputs.
        """
        if samples_number < 1:
            return []

        self._prepare_not_approximated_lossy_mixed_state(
            input_state[: self._hierarchy_level]  # Not approximated state part
        )

        self._prepare_approximated_input_state(
            input_state[self._hierarchy_level :]  # Approximated state part
        )

        number_of_samples_for_each_thread = self._compute_number_of_samples_for_each_thread(
            samples_number
        )

        # Context is required on Linux systems, as the default (fork) produces undesired
        # results! Spawn is default on osX and Windows and works as expected.
        multiprocessing_context = multiprocessing.get_context("spawn")

        with Pool(mp_context=multiprocessing_context) as p:
            samples_lists = p.map(
                self._simulate_in_parallel, number_of_samples_for_each_thread
            )

        samples = [sample for samples_list in samples_lists for sample in samples_list]

        return samples

    def _prepare_not_approximated_lossy_mixed_state(
        self, not_approximated_input_state_part: Sequence[int]
    ) -> None:
        """

        :param not_approximated_input_state_part:
        :return:
        """
        self._prepare_not_approximated_lossy_mixed_state_parts(
            not_approximated_input_state_part
        )
        self._prepare_not_approximated_lossy_mixed_state_parts_weights()

    def _prepare_not_approximated_lossy_mixed_state_parts(
        self, not_approximated_input_state_part: Sequence[int]
    ) -> None:
        """

        :param not_approximated_input_state_part:
        :return:
        """
        self._not_approximated_lossy_mixed_state_parts = []
        for number_of_particles_left in range(
            sum(not_approximated_input_state_part) + 1
        ):
            self._not_approximated_lossy_mixed_state_parts.extend(
                generate_lossy_n_particle_input_states(
                    not_approximated_input_state_part, number_of_particles_left
                )
            )

    def _prepare_not_approximated_lossy_mixed_state_parts_weights(self) -> None:
        """

        :return:
        """
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
                possible_weights[sum(state_part)] / binom(n, sum(state_part))
            )

            # In the case of binned input we have to also consider multiplicities.
            multiplicity_weight: float = 1
            for i in range(len(state_part)):
                multiplicity_weight *= binom(
                    self._not_approximated_lossy_mixed_state_parts[-1][i], state_part[i]
                )

            self._not_approximated_lossy_mixed_state_parts_weights[
                -1
            ] *= multiplicity_weight

    def _get_possible_lossy_inputs_weights(
        self, input_state: Sequence[int]
    ) -> List[float]:
        """

        :param input_state:
        :return:
        """
        weights = []

        # I'll use the same notation as in [1], for readability.
        n = int(sum(input_state))  # Initial number of particles.
        eta = self._uniform_transmissivity
        for l in range(n + 1):
            # l denotes number of particles left in the state
            weights.append(binom(n, l) * eta ** l * (1 - eta) ** (n - l))

        return weights

    def _prepare_approximated_input_state(
        self, approximated_input_state_part: Sequence[int]
    ) -> None:
        """

        :param approximated_input_state_part:
        :return:
        """
        # Assume exact simulation if hierarchy level is not specified.
        if not 0 <= self._hierarchy_level < len(self._permanent_calculator.matrix):
            self._approximated_input_state_part_possibilities = [[]]
            self._approximated_input_state_part_possibilities_weights = [1]
            return

        self._prepare_approximated_input_state_parts(approximated_input_state_part)
        self._prepare_approximated_input_state_parts_weights()

    def _prepare_approximated_input_state_parts(
        self, approximated_input_state_part: Sequence[int]
    ) -> None:
        """

        :param approximated_input_state_part:
        :return:
        """
        self._approximated_input_state_part_possibilities = []
        for number_of_particles_left in range(
            int(sum(approximated_input_state_part)) + 1
        ):
            state_part_possibility = zeros_like(approximated_input_state_part)
            state_part_possibility[0] = number_of_particles_left
            self._approximated_input_state_part_possibilities.append(
                state_part_possibility
            )

    def _prepare_approximated_input_state_parts_weights(self) -> None:
        """
        Prepare the probabilities of obtaining a given number of particles in the
        approximated part of the input.
        """
        self._approximated_input_state_part_possibilities_weights = self._get_possible_lossy_inputs_weights(
            self._approximated_input_state_part_possibilities[
                -1
            ]  # Last part contains all possible particles.
        )

    @staticmethod
    def _distribute_uniformly(values_number: int, bins: int) -> List[int]:
        """
        Uniformly distributes the values between the specified number of bins.

        TODO TR: Might be put in a more general file.

        :param values_number:
            The number of elements to be divided into bins.
        :param bins:
            The number of bins into which elements will be divided.

        :returns:
           The number of elements in each bin.
        """
        distributed_values = []

        for v in range(bins):
            distributed_values.append(values_number // bins)

        for i in range(values_number % bins):
            distributed_values[i] += 1

        return distributed_values

    def _compute_number_of_samples_for_each_thread(
        self, samples_number: int
    ) -> List[int]:
        """
        Computes the number of samples that each thread should return.

        :param samples_number:
            The total number of samples.

        :return:
            A list of samples that should be returned by each thread.
        """
        return self._distribute_uniformly(samples_number, self._threads_number)

    def _simulate_in_parallel(self, samples_number: int = 1) -> List[Tuple[int, ...]]:
        """
        This method produces given number of samples from lossy approximated
        (separable) state. It's meant to be run in parallel if so desired.

        :param samples_number:
            A number of samples returned by the method.

        :returns:
            A list of samples.
        """
        samples = []

        helper_strategy = GeneralizedCliffordsBSimulationStrategy(
            deepcopy(self._permanent_calculator)
        )

        for _ in range(samples_number):
            lossy_input = self._get_input_state_for_sampling()
            approximate_sampling_matrix = self._get_matrix_for_approximate_sampling()
            helper_strategy.set_new_matrix(approximate_sampling_matrix)
            samples.append(helper_strategy.simulate(lossy_input)[0])

        return samples

    def _get_input_state_for_sampling(self) -> ndarray:
        """
        Applies losses to the input state and returns the result.

        :return:
            Lossy input state.
        """
        approximated_part = self._approximated_input_state_part_possibilities[
            choice(
                range(len(self._approximated_input_state_part_possibilities)),
                p=self._approximated_input_state_part_possibilities_weights,
            )
        ]

        not_approximated_part = self._not_approximated_lossy_mixed_state_parts[
            choice(
                range(len(self._not_approximated_lossy_mixed_state_parts)),
                p=self._not_approximated_lossy_mixed_state_parts_weights,
            )
        ]
        return array(hstack([not_approximated_part, approximated_part]), dtype=int)

    def _permuted_interferometer_matrix(self) -> ndarray:
        """
        Permute the columns of the matrix for better symmetrization and possibly
        more accurate sampling.

        :return:
            The interferometer matrix with permuted columns.
        """
        columns_arrangement = arange(
            len(self._permanent_calculator.matrix)
        )  # We work with unitary matrices.
        permutation = arange(
            len(self._permanent_calculator.matrix) - self._hierarchy_level
        )
        shuffle(permutation)
        columns_arrangement[0 : len(permutation)] = permutation

        return array(self._permanent_calculator.matrix)[:, columns_arrangement]

    def _get_matrix_for_approximate_sampling(self) -> ndarray:
        """
        Prepares the matrix for the approximate sampling, as intended in [2].

        :return:
            A matrix for the approximate sampling.
        """
        random_phases_matrix = generate_random_phases_matrix_for_first_m_modes(
            len(self._qft_matrix) - self._hierarchy_level, len(self._qft_matrix)
        )

        return (
            self._permuted_interferometer_matrix()
            @ random_phases_matrix
            @ self._qft_matrix
        )
