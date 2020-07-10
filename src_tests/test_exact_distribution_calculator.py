import unittest
from numpy import array, std

from src.ExactLossyBosonSamplingDistributionCalculator \
    import ExactLossyBosonSamplingDistributionCalculator, BosonSamplingExperimentConfiguration


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

        self.initial_state = [1, 1, 1, 0, 0]

        # Create configuration object.
        self.experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.permutation_matrix,
            possible_outcomes=[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]],
            initial_state=self.initial_state,
            number_of_modes=len(self.initial_state),
            initial_number_of_particles=sum(self.initial_state),
            number_of_particles_lost=self.number_of_particles_lost,
            number_of_particles_left=sum(self.initial_state) - self.number_of_particles_lost
        )

    def test_probabilities_sum(self) -> None:
        exact_distribution_calculator = ExactLossyBosonSamplingDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()
        self.assertAlmostEqual(sum(exact_distribution), 1.0, delta=1e-4)

    def test_probabilities_standard_deviation(self) -> None:
        # Given that in this setup we require that probabilities of each outcome are equal, the standard deviation
        # should be close to 0.
        exact_distribution_calculator = ExactLossyBosonSamplingDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()
        standard_deviation = std(exact_distribution)
        self.assertAlmostEqual(standard_deviation, 0, delta=1e-4)
