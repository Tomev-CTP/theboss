__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to provide the tests for the distribution calculators.
"""

from scipy.stats import unitary_group
import unittest
from numpy import isclose

from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithUniformLosses, BSDistributionCalculatorWithFixedLosses,
    BosonSamplingExperimentConfiguration
)

from theboss.boson_sampling_utilities.permanent_calculators.chin_huh_permanent_calculator import ChinHuhPermanentCalculator


class TestDistributionCalculators(unittest.TestCase):

    def setUp(self) -> None:
        self._m: int = 4  # Modes number

        self._uniform_transmissivity: float = 0.5
        self._number_of_particles_lost = 2

        self._std_input = [1, 1, 1, 0]
        self._binned_input = [3, 1, 2, 0]

        self._matrix = unitary_group.rvs(self._m)

        self._permanent_calculator = ChinHuhPermanentCalculator(self._matrix)

        self._config: BosonSamplingExperimentConfiguration = None

    def _prepare_binned_input_test_setup(self) -> None:
        self._prepare_test_setup(self._binned_input)

    def _prepare_std_input_test_setup(self) -> None:
        self._prepare_test_setup(self._std_input)

    def _prepare_test_setup(self, input_state) -> None:
        self._config = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._matrix, initial_state=input_state,
            initial_number_of_particles=sum(input_state),
            number_of_modes=self._m,
            number_of_particles_lost=self._number_of_particles_lost,
            number_of_particles_left=sum(input_state) - self._number_of_particles_lost,
            uniform_transmissivity=self._uniform_transmissivity
        )
        self._permanent_calculator.input_state = input_state

    def test_uniform_losses_calc_distribution_sum_for_standard_input(self) -> None:
        self._prepare_std_input_test_setup()
        calc = BSDistributionCalculatorWithUniformLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()

        self.assertTrue(isclose(sum(distribution), 1))

    def test_fixed_losses_calc_distribution_sum_for_standard_input(self) -> None:
        self._prepare_std_input_test_setup()
        calc = BSDistributionCalculatorWithFixedLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(isclose(sum(distribution), 1))

    def test_uniform_losses_calc_distribution_sum_for_binned_input(self) -> None:
        self._prepare_binned_input_test_setup()
        calc = BSDistributionCalculatorWithUniformLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(isclose(sum(distribution), 1))

    def test_fixed_losses_calc_distribution_sum_for_binned_input(self) -> None:
        self._prepare_binned_input_test_setup()
        calc = BSDistributionCalculatorWithFixedLosses(
            self._config, self._permanent_calculator
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(isclose(sum(distribution), 1))
