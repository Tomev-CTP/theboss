__author__ = 'Tomasz Rybotycki'

import unittest
from math import factorial

from numpy import array, average, log2, sqrt
from numpy.random import randint
from scipy.special import binom

from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration, \
    BosonSamplingWithFixedLossesExactDistributionCalculator
from src.Quantum_Computations_Utilities import count_total_variation_distance, \
    count_tv_distance_error_bound_of_experiment_results, generate_haar_random_unitary_matrix
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy
from src_tests.common_code_for_tests import ApproximateDistributionCalculator


class TestClassicalLossyBosonSamplingSimulator(unittest.TestCase):

    def setUp(self) -> None:
        # Generate permutation matrix and define initial state.
        self.initial_state = [1, 1, 1, 1, 0]

        self.permutation_matrix = array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ])

        # Define some additional variables for more clear experiment configuration assignment.
        self.number_of_particles_lost = 2

        # Initialize experiment configuration.
        self.experiment_configuration = None

        # Define variables for bound calculation.
        self.number_of_samples_for_distribution_approximation = 1000
        self.error_probability_of_distance_bound = 0.001

    def test_approximate_and_exact_distribution_distance(self) -> None:
        self.experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.permutation_matrix,
            initial_state=self.initial_state,
            initial_number_of_particles=sum(self.initial_state),
            number_of_modes=len(self.initial_state),
            number_of_particles_lost=self.number_of_particles_lost,
            number_of_particles_left=sum(self.initial_state) - self.number_of_particles_lost
        )

        distance_between_estimate_and_ideal = self.__count_distance_between_approximate_and_exact_distribution()
        distance_bound_on_experiments_and_estimate = self.__count_distributions_tv_distance_bound()
        distance_bound_on_experiments_and_ideal = self.__calculate_distribution_error_bound()

        self.assertLessEqual(distance_between_estimate_and_ideal, distance_bound_on_experiments_and_estimate
                             + distance_bound_on_experiments_and_ideal)

    def __count_distance_between_approximate_and_exact_distribution(self) -> float:
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()

        strategy = FixedLossSimulationStrategy(self.experiment_configuration.interferometer_matrix,
                                               self.experiment_configuration.number_of_particles_left,
                                               self.experiment_configuration.number_of_modes)

        distribution_calculator = ApproximateDistributionCalculator(self.experiment_configuration, strategy)
        approximate_distribution = distribution_calculator.calculate_approximate_distribution()

        return count_total_variation_distance(exact_distribution, approximate_distribution)

    def __count_distributions_tv_distance_bound(self) -> float:
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)
        number_of_outcomes = len(exact_distribution_calculator.get_outcomes_in_proper_order())

        experiments_tv_distance_error_bound = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes, samples_number=self.number_of_samples_for_distribution_approximation,
            error_probability=self.error_probability_of_distance_bound
        )
        return experiments_tv_distance_error_bound

    def __calculate_distribution_error_bound(self) -> float:
        """
            Calculates error bound between the ideal and the experimental results. This is capital Delta
            of Oszmaniec and Brod papers.
        :return: Bound on distance between ideal and experimental results.
        """
        n = self.experiment_configuration.initial_number_of_particles
        l = self.experiment_configuration.number_of_particles_left
        error_bound = 1.0 - (float(factorial(n)) / (pow(n, l) * factorial(n - l)))
        return error_bound

    def test_approximate_and_exact_distribution_distance_for_haar_random_matrix(self) -> None:
        number_of_modes = 8
        initial_state = [1, 1, 1, 1, 0, 0, 0, 0]
        initial_number_of_particles = sum(initial_state)
        number_of_particles_lost = 2
        number_of_particles_left = initial_number_of_particles - number_of_particles_lost
        number_of_outcomes = binom(number_of_modes + number_of_particles_left - 1, number_of_particles_left)

        haar_random_matrices_number = 10 ** 2

        error_bound = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes, samples_number=haar_random_matrices_number,
            error_probability=self.error_probability_of_distance_bound
        )

        probabilities_list = []

        for i in range(haar_random_matrices_number):

            print(f'Current Haar random matrix index: {i} out of {haar_random_matrices_number}.')

            haar_random_matrix = generate_haar_random_unitary_matrix(number_of_modes)

            self.experiment_configuration = BosonSamplingExperimentConfiguration(
                interferometer_matrix=haar_random_matrix,
                initial_state=initial_state,
                initial_number_of_particles=initial_number_of_particles,
                number_of_modes=number_of_modes,
                number_of_particles_lost=number_of_particles_lost,
                number_of_particles_left=number_of_particles_left,
                probability_of_uniform_loss=0.2
            )

            strategy = FixedLossSimulationStrategy(self.experiment_configuration.interferometer_matrix,
                                                   self.experiment_configuration.number_of_particles_left,
                                                   self.experiment_configuration.number_of_modes)

            distribution_calculator = ApproximateDistributionCalculator(self.experiment_configuration, strategy)
            current_probabilities = distribution_calculator.calculate_approximate_distribution()

            if len(probabilities_list) == 0:
                probabilities_list = [[] for _ in range(len(current_probabilities))]

            for j in range(len(current_probabilities)):
                probabilities_list[j].append(current_probabilities[j])

        # Every probability should have probability 1 over outcomes number, if number of haar random matrices
        # goes to infinity. In that case I can select any probability.
        random_outcome_index = randint(0, len(current_probabilities))
        self.assertAlmostEqual(number_of_outcomes ** (-1), average(probabilities_list[random_outcome_index]),
                               delta=error_bound)
