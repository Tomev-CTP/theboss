__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to provide the tests for the distribution calculators.
"""

from scipy.stats import unitary_group
import unittest
from numpy import isclose

from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithUniformLosses,
    BSDistributionCalculatorWithFixedLosses,
    BosonSamplingExperimentConfiguration,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)


class TestDistributionCalculators(unittest.TestCase):
    def setUp(self) -> None:
        self._m: int = 4  # Modes number

        self._uniform_transmission_probability: float = 0.5
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
            uniform_transmission_probability=self._uniform_transmission_probability,
        )
        self._permanent_calculator.input_state = input_state

    def test_uniform_losses_calc_distribution_sum_for_standard_input(self) -> None:
        """
        Test uniform losses calculator for standard input.
        """
        self._prepare_test_setup(self._std_input)
        calc = BSDistributionCalculatorWithUniformLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()

        self.assertTrue(isclose(sum(distribution), 1))

    def test_fixed_losses_calc_distribution_sum_for_standard_input(self) -> None:
        """
        Test fixed losses calculator for standard input.
        """
        self._prepare_test_setup(self._std_input)
        calc = BSDistributionCalculatorWithFixedLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(isclose(sum(distribution), 1))

    def test_uniform_losses_calc_distribution_sum_for_binned_input(self) -> None:
        """
        Test uniform losses calculator for binned input.
        """
        self._prepare_test_setup(self._binned_input)
        calc = BSDistributionCalculatorWithUniformLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(isclose(sum(distribution), 1))

    def test_fixed_losses_calc_distribution_sum_for_binned_input(self) -> None:
        """
        Test fixed losses calculator for standard input.
        """
        self._prepare_test_setup(self._binned_input)
        calc = BSDistributionCalculatorWithFixedLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(isclose(sum(distribution), 1))
