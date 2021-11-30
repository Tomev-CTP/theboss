__author__ = "Tomasz Rybotycki"

import unittest

from numpy import array, complex128, int64, allclose

from tests import (BSPermanentCalculatorFactory, PermanentCalculatorType, BosonSamplingExperimentConfiguration,
                   BSDistributionCalculatorWithUniformLosses, BSDistributionCalculatorWithFixedLosses)


class TestExactLossyBosonSamplingDistributionCalculator(unittest.TestCase):

    def setUp(self) -> None:
        # Define some additional variables for more clear experiment configuration assignment.
        self.number_of_particles_lost = 2

        # Generate permutation matrix and define initial state.
        self.permutation_matrix = array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ], dtype=complex128)

        self.initial_state = array([1, 1, 1, 0, 0], dtype=int64)

        # Create configuration object.
        self.experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.permutation_matrix,
            initial_state=self.initial_state,
            number_of_modes=len(self.initial_state),
            initial_number_of_particles=sum(self.initial_state),
            number_of_particles_lost=self.number_of_particles_lost,
            number_of_particles_left=sum(self.initial_state) - self.number_of_particles_lost,
            uniform_transmissivity=0.8
        )

        self._calculator_type = PermanentCalculatorType.PARALLEL_CHIN_HUH
        self._permanent_calculator_factory = BSPermanentCalculatorFactory(matrix=None, input_state=None,
                                                                          output_state=None,
                                                                          calculator_type=self._calculator_type)
        self._permanent_calculator = self._permanent_calculator_factory.generate_calculator()

    def test_probabilities_sum_in_fixed_losses_scenario(self) -> None:
        exact_distribution_calculator = \
            BSDistributionCalculatorWithFixedLosses(self.experiment_configuration,
                                                    self._permanent_calculator)
        exact_distribution = exact_distribution_calculator.calculate_distribution()
        self.assertAlmostEqual(sum(exact_distribution), 1.0, delta=1e-4)

    def test_probabilities_sum_in_uniform_losses_scenario(self) -> None:
        exact_distribution_calculator = \
            BSDistributionCalculatorWithUniformLosses(self.experiment_configuration,
                                                      self._permanent_calculator)
        exact_distribution = exact_distribution_calculator.calculate_distribution()
        self.assertAlmostEqual(sum(exact_distribution), 1.0, delta=1e-4)

    def test_probabilities(self) -> None:
        exact_distribution_calculator = \
            BSDistributionCalculatorWithUniformLosses(self.experiment_configuration,
                                                      self._permanent_calculator)
        exact_distribution = exact_distribution_calculator.calculate_distribution()
        self.assertTrue(
            allclose(
                exact_distribution,
                [
                    0.008, 0.032, 0.0, 0.0, 0.032, 0.032, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.128, 0.0, 0.0, 0.0, 0.128, 0.0, 0.0, 0.128, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.512, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
            )
        )
