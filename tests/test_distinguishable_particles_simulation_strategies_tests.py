__author__ = "Tomasz Rybotycki"

"""
    This script contains test for the distinguishable BS simulation strategies.
"""

import unittest
from scipy.stats import unitary_group
from numpy import diag, array
from numpy.random import random
from typing import List, Sequence, Tuple

from theboss.boson_sampling_utilities import (
    bosonic_space_dimension,
    generate_possible_states,
)

from theboss.quantum_computations_utilities import (
    compute_minimal_number_of_samples_for_desired_accuracy,
    count_total_variation_distance,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)

from theboss.simulation_strategies.distinguishable_particles_simulation_strategy import (
    DistinguishableParticlesSimulationStrategy,
    SimulationStrategyInterface,
)

from theboss.simulation_strategies.distinguishable_particles_uniform_losses_simulation_strategy import (
    DistinguishableParticlesUniformLossesSimulationStrategy,
)

from theboss.simulation_strategies.distinguishable_particles_nonuniform_losses_simulation_strategy import (
    DistinguishableParticlesNonuniformLossesSimulationStrategy,
)

from theboss.distribution_calculators.fixed_losses_distinguishable_particles_distribution_calculator import (
    FixedLossesDistinguishableParticlesDistributionCalculator,
    BSDistributionCalculatorInterface,
    BosonSamplingExperimentConfiguration,
)

from theboss.distribution_calculators.uniform_losses_distinguishable_particles_distribution_calculator import (
    UniformLossesDistinguishableParticlesDistributionCalculator,
)

from theboss.distribution_calculators.nonuniform_losses_distinguishable_particles_distribution_calculator import (
    NonUniformlyLossyDistinguishableParticlesDistributionCalculator,
)


class TestDistinguishableParticlesSamplingSimulationStrategies(unittest.TestCase):

    """
    Unit tests class for testing the accuracy of samplers for the distribution of
    distinguishable particles and different losses.
    """

    def setUp(self) -> None:

        self._modes_number: int = 4
        self._particles_number: int = 4

        self._std_input: List[int] = [1 for _ in range(self._modes_number)]

        self._binned_input: List[int] = [0 for _ in range(self._modes_number)]
        self._binned_input[0] = self._modes_number - 1
        self._binned_input[1] = 1

        self._nonuniform_losses: List[float] = random(self._modes_number)

        self._error_probability: float = 0.01
        self._statistical_upper_bound: float = 0.1

        self._matrix: Sequence[Sequence[complex]] = unitary_group.rvs(
            self._modes_number
        )

        self._uniform_transmissivity: float = 0.4

        self._config: BosonSamplingExperimentConfiguration
        self._config = BosonSamplingExperimentConfiguration(
            self._matrix,
            self._std_input,
            self._particles_number,
            self._modes_number,
            0,
            self._particles_number,
            1,
        )

        self._required_samples_number: int = compute_minimal_number_of_samples_for_desired_accuracy(
            bosonic_space_dimension(self._particles_number, self._modes_number),
            self._error_probability,
            self._statistical_upper_bound,
        )

        self._possible_states: List[Tuple[int, ...]] = generate_possible_states(
            self._particles_number, self._modes_number
        )

        self._simulator: SimulationStrategyInterface
        self._simulator = DistinguishableParticlesSimulationStrategy(self._matrix)

        self._permanent_calculator: RyserPermanentCalculator
        self._permanent_calculator = RyserPermanentCalculator(
            self._matrix, self._std_input
        )

        self._distribution_calculator: BSDistributionCalculatorInterface
        self._distribution_calculator = FixedLossesDistinguishableParticlesDistributionCalculator(
            self._config, self._permanent_calculator
        )

    def _prepare_experiment_with_bunched_input(self) -> None:
        """
        Prepares the setup for the binned output simulation.
        """
        self._config.initial_state = self._binned_input
        self._distribution_calculator.configuration.initial_state = self._binned_input

    def _prepare_uniformly_lossy_experiment(self) -> None:
        """
        Sets up the uniformly lossy simulation experiment.
        """
        self._possible_states = generate_possible_states(
            self._particles_number, self._modes_number, True
        )
        self._required_samples_number = compute_minimal_number_of_samples_for_desired_accuracy(
            len(self._possible_states),
            self._error_probability,
            self._statistical_upper_bound,
        )
        self._config.uniform_transmissivity = self._uniform_transmissivity
        self._distribution_calculator = UniformLossesDistinguishableParticlesDistributionCalculator(
            self._config, self._permanent_calculator
        )

    def _prepare_nonuniformly_lossy_experiment(self) -> None:
        """
        Sets up the non-uniformly lossy simulation experiment.
        """
        self._prepare_uniformly_lossy_experiment()
        self._simulator = DistinguishableParticlesNonuniformLossesSimulationStrategy(
            array(self._matrix) @ diag(self._nonuniform_losses)
        )

        self._distribution_calculator = NonUniformlyLossyDistinguishableParticlesDistributionCalculator(
            array(self._matrix) @ diag(self._nonuniform_losses),
            self._config.initial_state,
        )

    def _get_frequencies(self) -> List[float]:
        """
        Draws samples from configured sampler and computes their frequencies.

        :return:
            Returns the frequencies of every output state for the sampler specified
            at the beginning of a test.
        """
        samples: List[Tuple[int, ...]] = self._simulator.simulate(
            self._config.initial_state, self._required_samples_number
        )

        frequencies: List[float] = []

        for state in self._possible_states:
            frequencies.append(samples.count(state) / self._required_samples_number)

        return frequencies

    def _check_sampler_accuracy(self):
        """
        This method is the general part of the tests. It computes distributions and
        frequencies, and checks if they are close enough, or not.
        """
        distribution: List[
            float
        ] = self._distribution_calculator.calculate_distribution()
        frequencies: List[float] = self._get_frequencies()

        tvd: float = count_total_variation_distance(distribution, frequencies)

        self.assertTrue(
            tvd < self._statistical_upper_bound,
            f"The tvd ({tvd}) is larger than the "
            f"statistical bound ({self._statistical_upper_bound}).",
        )

    def test_lossless_sampling_with_standard_input(self) -> None:
        """
        Test the accuracy of exact sampler for distinguishable particles and standard
        input.
        """
        self._check_sampler_accuracy()

    def test_lossless_sampling_with_binned_input(self) -> None:
        """
        Test the accuracy of exact sampler for distinguishable particles and binned
        input.
        """
        self._prepare_experiment_with_bunched_input()

        self._check_sampler_accuracy()

    def test_uniform_losses_strategy_with_lossless_standard_input(self) -> None:
        """
        Test the accuracy of sampler for uniformly lossy interferometers with
        standard input and no losses.
        """
        self._simulator = DistinguishableParticlesUniformLossesSimulationStrategy(
            self._matrix, self._config.uniform_transmissivity
        )

        self._check_sampler_accuracy()

    def test_uniform_losses_sampling_with_lossy_standard_input(self) -> None:
        """
        Test the accuracy of sampler for uniformly lossy interferometers with
        standard input and losses.
        """
        self._prepare_uniformly_lossy_experiment()
        self._simulator = DistinguishableParticlesUniformLossesSimulationStrategy(
            self._matrix, self._config.uniform_transmissivity
        )

        self._check_sampler_accuracy()

    def test_uniform_losses_sampling_with_lossless_binned_input(self) -> None:
        """
        Test the accuracy of sampler for uniformly lossy interferometers with
        binned input and no losses.
        """
        self._prepare_experiment_with_bunched_input()
        self._simulator = DistinguishableParticlesUniformLossesSimulationStrategy(
            self._matrix, self._config.uniform_transmissivity
        )

        self._check_sampler_accuracy()

    def test_uniform_losses_sampling_with_lossy_binned_input(self) -> None:
        """
        Test the accuracy of sampler for uniformly lossy interferometers with
        binned input and losses.
        """
        self._prepare_experiment_with_bunched_input()
        self._prepare_uniformly_lossy_experiment()
        self._simulator = DistinguishableParticlesUniformLossesSimulationStrategy(
            self._matrix, self._config.uniform_transmissivity
        )

        self._check_sampler_accuracy()

    def test_nonuniform_losses_sampling_with_losses(self) -> None:
        """
        Test the accuracy of sampler for non-uniformly lossy interferometers with
        standard input.
        """
        self._prepare_nonuniformly_lossy_experiment()
        self._check_sampler_accuracy()

    def test_nonuniform_losses_sampling_with_lossy_binned_input(self) -> None:
        """
        Test the accuracy of sampler for non-uniformly lossy interferometers with
        binned input.
        """
        self._prepare_experiment_with_bunched_input()
        self._prepare_nonuniformly_lossy_experiment()
        self._check_sampler_accuracy()
