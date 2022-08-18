__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to provide the tests for the distribution calculators for
    BS with distinguishable particles.
"""

from scipy.stats import unitary_group
import unittest
from numpy import isclose, diag
from numpy.random import random

from theboss.distribution_calculators.nonuniform_losses_distinguishable_particles_distribution_calculator import (
    NonUniformlyLossyDistinguishableParticlesDistributionCalculator,
)

from theboss.distribution_calculators.uniform_losses_distinguishable_particles_distribution_calculator import (
    UniformLossesDistinguishableParticlesDistributionCalculator,
    FixedLossesDistinguishableParticlesDistributionCalculator,
    BosonSamplingExperimentConfiguration,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)


class TestDistributionCalculators(unittest.TestCase):
    def setUp(self) -> None:
        self._m: int = 4  # Modes number

        self._uniform_transmissivity: float = 0.5
        self._number_of_particles_lost = 2

        self._std_input = [1, 1, 1, 0]
        self._binned_input = [3, 1, 2, 0]

        self._matrix = unitary_group.rvs(self._m)

        self._permanent_calculator = RyserPermanentCalculator(self._matrix)

        self._config: BosonSamplingExperimentConfiguration

    def _prepare_test_setup(self, input_state) -> None:
        """
        Boilerplate code for preparing the experiment configuration.

        :param input_state:
            Input state (in 2nd quantization representation).
        """
        self._config = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._matrix,
            initial_state=input_state,
            number_of_particles_lost=self._number_of_particles_lost,
            uniform_transmissivity=self._uniform_transmissivity,
        )
        self._permanent_calculator.input_state = input_state

    def test_uniform_losses_calc_distribution_sum_for_standard_input(self) -> None:
        """
        Test uniform losses calculator for standard input.
        """
        self._prepare_test_setup(self._std_input)
        calc = UniformLossesDistinguishableParticlesDistributionCalculator(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()

        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_fixed_losses_calc_distribution_sum_for_standard_input(self) -> None:
        """
        Test fixed losses calculator for standard input.
        """
        self._prepare_test_setup(self._std_input)
        calc = FixedLossesDistinguishableParticlesDistributionCalculator(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_uniform_losses_calc_distribution_sum_for_binned_input(self) -> None:
        """
        Test uniform losses calculator for binned input.
        """
        self._prepare_test_setup(self._binned_input)
        calc = UniformLossesDistinguishableParticlesDistributionCalculator(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_fixed_losses_calc_distribution_sum_for_binned_input(self) -> None:
        """
        Test fixed losses calculator for binned input.
        """
        self._prepare_test_setup(self._binned_input)
        calc = FixedLossesDistinguishableParticlesDistributionCalculator(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_nonuniform_losses_calc_for_standard_input(self) -> None:
        """
        Test non-uniform losses calculator for standard input.
        """
        self._prepare_test_setup(self._std_input)
        calc = NonUniformlyLossyDistinguishableParticlesDistributionCalculator(
            self._matrix @ diag(random(self._m)), self._std_input
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_nonuniform_losses_calc_for_binned_input(self) -> None:
        """
        Test non-uniform losses calculator for binned input.
        """
        self._prepare_test_setup(self._binned_input)
        calc = NonUniformlyLossyDistinguishableParticlesDistributionCalculator(
            self._matrix @ diag(random(self._m)), self._binned_input
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )
