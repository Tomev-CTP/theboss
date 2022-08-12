__author__ = "Tomasz Rybotycki"

"""
    This script contains test for the uniform sampling strategies.
"""

import unittest
from typing import List, Tuple

from theboss.boson_sampling_utilities import generate_possible_states

from theboss.quantum_computations_utilities import (
    compute_minimal_number_of_samples_for_desired_accuracy,
    count_total_variation_distance,
)


from theboss.simulation_strategies.uniform_sampler import UniformSamplingStrategy
from theboss.simulation_strategies.uniform_sampler_for_uniform_losses import (
    SimulationStrategyInterface,
    UniformSamplingWithUniformLossesStrategy,
)
from theboss.simulation_strategies.uniform_sampler_from_lossy_outputs import (
    UniformSamplingFromLossyOutputsStrategy,
)

from theboss.distribution_calculators.uniform_losses_uniform_distribution_calculator import (
    BSDistributionCalculatorInterface,
    UniformDistributionCalculatorForUniformlyLossyBS,
)
from theboss.distribution_calculators.uniform_distribution_calculator_for_lossy_bs import (
    UniformDistributionCalculatorForLossyBS,
)


class TestUniformSamplingSimulationStrategies(unittest.TestCase):

    """
    Unit tests class for testing the accuracy of samplers for the uniform distribution
    over (possibly) lossy outputs.
    """

    def setUp(self) -> None:

        self._modes_number: int = 4
        self._particles_number: int = 4

        self._std_input: List[int] = [1 for _ in range(self._modes_number)]

        self._binned_input: List[int] = [0 for _ in range(self._modes_number)]
        self._binned_input[0] = self._modes_number - 1
        self._binned_input[1] = 1

        self._test_state: List[int] = self._std_input

        self._error_probability: float = 0.01
        self._statistical_upper_bound: float = 0.1

        self._uniform_transmissivity: float = 0.4

        self._possible_states = generate_possible_states(
            self._particles_number, self._modes_number, False
        )

        self._required_samples_number: int = compute_minimal_number_of_samples_for_desired_accuracy(
            len(self._possible_states),
            self._error_probability,
            self._statistical_upper_bound,
        )

        self._simulator: SimulationStrategyInterface
        self._simulator = UniformSamplingStrategy()

        self._distribution_calculator: BSDistributionCalculatorInterface
        self._distribution_calculator = UniformDistributionCalculatorForUniformlyLossyBS(
            self._modes_number, self._particles_number, 1
        )

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
        self._distribution_calculator.transmissivity = self._uniform_transmissivity

    def _prepare_nonuniformly_lossy_experiment(self) -> None:
        """
        Sets up the non-uniformly lossy simulation experiment.
        """
        self._prepare_uniformly_lossy_experiment()
        self._simulator = UniformSamplingFromLossyOutputsStrategy()

        self._distribution_calculator = UniformDistributionCalculatorForLossyBS(
            self._modes_number, self._particles_number
        )

    def _get_frequencies(self) -> List[float]:
        """
        Draws samples from configured sampler and computes their frequencies.

        :return:
            Returns the frequencies of every output state for the sampler specified
            at the beginning of a test.
        """
        samples: List[Tuple[int, ...]] = self._simulator.simulate(
            self._test_state, self._required_samples_number
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
        ] = self._distribution_calculator.calculate_probabilities_of_outcomes(
            self._possible_states
        )
        frequencies: List[float] = self._get_frequencies()

        tvd: float = count_total_variation_distance(distribution, frequencies)

        self.assertTrue(
            tvd < self._statistical_upper_bound,
            f"The tvd ({tvd}) is larger than the "
            f"statistical bound ({self._statistical_upper_bound}).",
        )

    def test_uniform_sampler_with_standard_input(self) -> None:
        """
        Test the accuracy of uniform sampler for standard input.
        """
        self._check_sampler_accuracy()

    def test_uniform_sampler_with_binned_input(self) -> None:
        """
        Test the accuracy of uniform sampler for binned input.
        """
        self._test_state = self._binned_input

        self._check_sampler_accuracy()

    def test_uniformly_lossy_uniform_sampling_with_lossless_standard_input(
        self,
    ) -> None:
        """
        Test the accuracy of uniformly lossy uniform sampler for lossless standard
        input state.
        """
        self._simulator = UniformSamplingWithUniformLossesStrategy()
        self._check_sampler_accuracy()

    def test_uniform_losses_sampling_with_lossy_standard_input(self) -> None:
        """
        Test the accuracy of uniformly lossy uniform sampler for uniformly lossy
        standard input state.
        """
        self._prepare_uniformly_lossy_experiment()
        self._simulator = UniformSamplingWithUniformLossesStrategy(
            self._uniform_transmissivity
        )

        self._check_sampler_accuracy()

    def test_uniform_losses_sampling_with_lossless_binned_input(self) -> None:
        """
        Test the accuracy of uniformly lossy uniform sampler for lossless binned
        input state.
        """
        self._test_state = self._binned_input
        self._simulator = UniformSamplingWithUniformLossesStrategy()

        self._check_sampler_accuracy()

    def test_uniform_losses_sampling_with_lossy_binned_input(self) -> None:
        """
        Test the accuracy of uniformly lossy uniform sampler for uniformly lossy
        binned input state.
        """
        self._test_state = self._binned_input
        self._prepare_uniformly_lossy_experiment()
        self._simulator = UniformSamplingWithUniformLossesStrategy(
            self._uniform_transmissivity
        )

        self._check_sampler_accuracy()

    def test_nonuniform_losses_sampling_with_losses(self) -> None:
        """
        Test the accuracy of uniform sampler from lossy states starting with
        binned input.
        """
        self._prepare_nonuniformly_lossy_experiment()
        self._check_sampler_accuracy()

    def test_nonuniform_losses_sampling_with_lossy_binned_input(self) -> None:
        """
        Test the accuracy of uniform sampler from lossy states starting with
        binned input.
        """
        self._test_state = self._binned_input
        self._prepare_nonuniformly_lossy_experiment()
        self._check_sampler_accuracy()
