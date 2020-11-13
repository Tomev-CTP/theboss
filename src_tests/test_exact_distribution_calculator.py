__author__ = 'Tomasz Rybotycki'

import unittest

from numpy import array

from src.LossyBosonSamplingExactDistributionCalculators import (
    BosonSamplingExperimentConfiguration,
    BosonSamplingWithFixedLossesExactDistributionCalculator,
    BosonSamplingWithUniformLossesExactDistributionCalculator)


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
        ])

        self.initial_state = array([1, 1, 1, 0, 0])

        # Create configuration object.
        self.experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.permutation_matrix,
            initial_state=self.initial_state,
            number_of_modes=len(self.initial_state),
            initial_number_of_particles=sum(self.initial_state),
            number_of_particles_lost=self.number_of_particles_lost,
            number_of_particles_left=sum(self.initial_state) - self.number_of_particles_lost,
            probability_of_uniform_loss=0.8
        )

    def test_probabilities_sum_in_fixed_losses_scenario(self) -> None:
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()
        self.assertAlmostEqual(sum(exact_distribution), 1.0, delta=1e-4)

    def test_probabilities_sum_in_uniform_losses_scenario(self) -> None:
        exact_distribution_calculator = \
            BosonSamplingWithUniformLossesExactDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()
        self.assertAlmostEqual(sum(exact_distribution), 1.0, delta=1e-4)
