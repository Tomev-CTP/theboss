__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to test non-uniform losses simulation strategies are
    accurate enough in the presence of such losses. Here, we focus only on the 
    simulations with NON-UNIFORM losses. Uniform losses and lossless simulations
    accuracy tests for these strategies have been placed in the other files.
"""

import unittest
from scipy.stats import unitary_group
from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    bosonic_space_dimension,
    generate_possible_states,
)
from theboss.quantum_computations_utilities import (
    compute_minimal_number_of_samples_for_desired_accuracy,
    count_total_variation_distance_dicts,
)
from typing import List, DefaultDict, Tuple, Dict
from numpy import ndarray, zeros_like, block, sqrt
from numpy.linalg import svd
from collections import defaultdict

from theboss.boson_sampling_utilities.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)

from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithUniformLosses,
    BosonSamplingExperimentConfiguration,
)

from theboss.simulation_strategies.simulation_strategy_factory import (
    StrategyType,
    SimulationStrategyFactory,
)


class TestNonuniformLossesStrategies(unittest.TestCase):
    """
    A class with unit test for nonuniform sampling simulations.
    """

    def setUp(self) -> None:

        # Statistical error settings.
        self._desired_statistical_accuracy: float = 0.1  # So the tests go quick(er).
        self._probability_of_error: float = 0.01

        self._m: int = 2  # Block matrix modes number.

        # Interferometer matrix preparation.
        self._blocks_transmissivities: List[float] = [0.5, 0.2]
        block_matrices: List[ndarray] = [unitary_group.rvs(self._m) for _ in range(2)]

        self._interferometer_matrix: ndarray = self._get_interferometer_matrix(
            block_matrices
        )

        # Compute distributions
        self._distribution: Dict[
            Tuple[int, ...], float
        ] = self._get_theoretical_distribution(block_matrices)

        self._initial_state: List[int] = [1 for _ in range(self._m * 2)]

        self._config: BosonSamplingExperimentConfiguration
        self._config = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._interferometer_matrix,
            initial_state=self._initial_state,
            initial_number_of_particles=self._m * 2,
            number_of_modes=self._m * 2,
            number_of_particles_lost=0,
            number_of_particles_left=self._m * 2,
            hierarchy_level=2 * self._m,
        )
        self._permanent_calculator: RyserPermanentCalculator = RyserPermanentCalculator(
            self._interferometer_matrix, self._initial_state, None
        )

        self._strategy_type: StrategyType = StrategyType.BOBS

    def _get_samples_number(self, modes_number: int) -> int:
        """
        Computes the minimal number of samples that is required to obtained desired
        (hard coded) accuracy with high (hard coded) probability. This takes into
        account only the statistical error.

        Notice that we don't have to take into account the whole :math:`2m \\times 2m`
        bosonic space, as the matrix we're analysing has a block form.

        :param modes_number:
            The number of modes (and also particles, since we're in the :math:`m = n`
            regime) that we consider in the experiment.

        :return:
            The minimal number of particles required to obtain specified accuracy with
            high probability.
        """

        return compute_minimal_number_of_samples_for_desired_accuracy(
            outcomes_number=bosonic_space_dimension(modes_number, modes_number, True)
            ** 2,
            error_probability=self._probability_of_error,
            expected_distance=self._desired_statistical_accuracy,
        )

    def _get_interferometer_matrix(self, block_matrices: List[ndarray]) -> ndarray:
        """
        Prepares matrices for the sampling. We do this by sampling two Haar-random
        matrices from the unitary group and applying (DIFFERENT) uniform losses to them.
        Then the interferometer matrix that we're interested in is in block form
        :math:`M = [[A, 0]^T, [0, B]^T]`.

        :param block_matrices:
            Two :math:`m \\times m` matrices sampled Haar-randomly from the unitary
            group.

        :return:
            A list of matrices that will be used for the simulations.
        """
        zeros_matrix: ndarray = zeros_like(block_matrices[0])
        return block(
            [
                [
                    block_matrices[0] * sqrt(self._blocks_transmissivities[0]),
                    zeros_matrix,
                ],
                [
                    zeros_matrix,
                    block_matrices[1] * sqrt(self._blocks_transmissivities[1]),
                ],
            ]
        )

    def _get_theoretical_distribution(
        self, block_matrices: List[ndarray]
    ) -> Dict[Tuple[int, ...], float]:
        """
        Computes distribution of large interferometer made from 2 smaller block
        interferometers with uniform losses.

        :param block_matrices:
            Already sampled block matrices.

        :return:
            Theoretical distribution for BS with specified matrix.
        """

        possible_states: List[Tuple[int, ...]] = generate_possible_states(
            self._m, self._m, losses=True
        )
        partial_distributions: List[List[float]] = []

        # Get partial distributions.
        for i in range(len(block_matrices)):

            transmissivity: float = self._blocks_transmissivities[i]

            block_input_state: List[int] = [1 for _ in range(self._m)]

            # Extract losses from the block matrix
            u: ndarray
            s: List[float]
            v: ndarray
            u, s, v = svd(block_matrices[i])
            lossless_block_matrix: ndarray = u @ v

            config: BosonSamplingExperimentConfiguration
            config = BosonSamplingExperimentConfiguration(
                lossless_block_matrix,
                block_input_state,
                self._m,
                self._m,
                0,
                self._m,
                transmissivity,
            )

            permanent_calculator = RyserPermanentCalculator(
                lossless_block_matrix, block_input_state, None
            )

            distribution_calculator = BSDistributionCalculatorWithUniformLosses(
                config, permanent_calculator
            )

            partial_distributions.append(
                distribution_calculator.calculate_probabilities_of_outcomes(
                    possible_states
                )
            )

        distribution: DefaultDict[Tuple[int, ...], float] = defaultdict(lambda: 0)

        # Join partial distributions.
        for i in range(len(possible_states)):
            for j in range(len(possible_states)):
                distribution[possible_states[i] + possible_states[j]] = (
                    partial_distributions[0][i] * partial_distributions[1][j]
                )

        return distribution

    @staticmethod
    def _compute_frequencies(
        counts: Dict[Tuple[int, ...], int]
    ) -> DefaultDict[Tuple[int, ...], float]:
        """
        Computes empirical frequencies from the counts.

        :param counts:
            State counts gathered during the experiments.

        :return:
            Empirical frequencies of the states.
        """
        frequencies: DefaultDict[Tuple[int, ...], float] = defaultdict(lambda: 0)
        samples_number: int = 0

        for state in counts:
            samples_number += counts[state]

        for state in counts:
            frequencies[state] = counts[state] / samples_number

        return frequencies

    def _compute_bobs_approximation_tvd_bound(self):
        """
        Compute bounds for BOBS strategy. For details check [2], formula (22).

        :return: TVD bound for BOBS algorithm.
        """
        eta_eff = max(self._blocks_transmissivities)
        n = 2 * self._m

        bound = pow(eta_eff, 2) / 2
        bound *= n - self._config.hierarchy_level
        bound += eta_eff * (1 - eta_eff) / 2

        return bound

    def _perform_accuracy_test(self, approximation_distance_bound: float = 0) -> None:
        """
        Boilerplate code for the non-uniform lossy strategies tests. It takes care of
        producing strategies, samples, frequencies and finally
        :param approximation_distance_bound:

        :return:
        """
        # Prepare strategy.
        strategy_factory: SimulationStrategyFactory = SimulationStrategyFactory(
            experiment_configuration=self._config,
            bs_permanent_calculator=self._permanent_calculator,
            strategy_type=self._strategy_type,
        )
        strategy = strategy_factory.generate_strategy()

        # Get samples.
        samples_number: int = self._get_samples_number(self._m)
        samples = strategy.simulate(self._initial_state, samples_number)

        # Get counts.
        counts: DefaultDict[Tuple[int, ...], int] = defaultdict(lambda: 0)

        for sample in samples:
            counts[sample] += 1

        # Get frequencies.
        frequencies: Dict[Tuple[int, ...], float] = self._compute_frequencies(counts)

        tvd: float = count_total_variation_distance_dicts(
            frequencies, self._distribution
        )

        self.assertTrue(
            tvd < self._desired_statistical_accuracy + approximation_distance_bound,
            f"TVD ({tvd}) is greater than expected ("
            f"{self._desired_statistical_accuracy + approximation_distance_bound})!\n\n"
            f"Tested matrix was: {self._interferometer_matrix}",
        )

    def test_lossy_net_gcc_accuracy(self) -> None:
        """
        Test accuracy of the non-uniform lossy net GCC Strategy in the presence of the
        non-uniform losses.
        """
        self._strategy_type = StrategyType.LOSSY_NET_GCC
        self._perform_accuracy_test()

    def test_exact_bobs_accuracy(self) -> None:
        """
        Test accuracy of the general BOBS Strategy in the presence of the non-uniform
        losses without approximations.
        """
        self._perform_accuracy_test()

    def test_small_approximation_bobs_accuracy(self) -> None:
        """
        Test accuracy of the general BOBS strategy in the presence of the non-uniform
        losses with small approximation.
        """
        self._config.hierarchy_level = 2
        approximation_bound: float = self._compute_bobs_approximation_tvd_bound()
        self._perform_accuracy_test(approximation_bound)

    def test_high_approximation_bobs_accuracy(self) -> None:
        """
        Test accuracy of the general BOBS strategy in the presence of the non-uniform
        losses with high approximation.
        """
        self._config.hierarchy_level = 1
        approximation_bound: float = self._compute_bobs_approximation_tvd_bound()
        self._perform_accuracy_test(approximation_bound)
