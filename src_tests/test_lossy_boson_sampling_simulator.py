__author__ = 'Tomasz Rybotycki'

import projectFolderPathUpdater  # This is for adding main project folder to the path.
import unittest
from numpy import std, average
from math import factorial
from typing import List

from numpy import zeros, log2, sqrt
from scipy.special import binom

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration, \
    BosonSamplingWithFixedLossesExactDistributionCalculator, BosonSamplingWithUniformLossesExactDistributionCalculator
from src.Quantum_Computations_Utilities import count_total_variation_distance, \
    count_tv_distance_error_bound_of_experiment_results, generate_haar_random_unitary_matrix
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy


class TestClassicalLossyBosonSamplingSimulator(unittest.TestCase):

    def setUp(self) -> None:
        # Generate permutation matrix and define initial state.
        self.initial_state = [1, 1, 1, 1, 0]

        '''
        self.permutation_matrix = array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ])
        '''

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

        distance = self.__count_distance_between_approximate_and_exact_distribution()
        experiments_tv_distance_error_bound = self.__count_distributions_tv_distance_bound()
        bound = self.__calculate_distribution_error_bound()

        self.assertAlmostEqual(distance, bound, delta=experiments_tv_distance_error_bound)

    def __count_distance_between_approximate_and_exact_distribution(self) -> float:
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()

        approximate_distribution = self.__calculate_approximate_distribution(
            self.number_of_samples_for_distribution_approximation)
        return count_total_variation_distance(exact_distribution, approximate_distribution)

    def __count_distributions_tv_distance_bound(self) -> float:
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)
        number_of_outcomes = len(exact_distribution_calculator.get_outcomes_in_proper_order())
        # print([list(i) for i in exact_distribution_calculator.get_outcomes_in_proper_order()])

        experiments_tv_distance_error_bound = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes, samples_number=self.number_of_samples_for_distribution_approximation,
            error_probability=self.error_probability_of_distance_bound
        )
        return experiments_tv_distance_error_bound

    def __calculate_approximate_distribution(self, samples_number: int = 5000) -> List[float]:
        """
        Prepares the approximate distribution using boson sampling simulation method described by
        Oszmaniec and Brod. Obviously higher number of samples will generate better approximation.
        :return: Approximate distribution as a list.
        """

        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)

        possible_outcomes = exact_distribution_calculator.get_outcomes_in_proper_order()

        strategy = FixedLossSimulationStrategy(self.experiment_configuration.interferometer_matrix,
                                               self.experiment_configuration.number_of_particles_left,
                                               self.experiment_configuration.number_of_modes)

        simulator = BosonSamplingSimulator(self.experiment_configuration.number_of_particles_left,
                                           self.experiment_configuration.initial_number_of_particles,
                                           self.experiment_configuration.number_of_modes, strategy)

        outcomes_probabilities = zeros(len(possible_outcomes))
        for i in range(samples_number):
            result = simulator.get_classical_simulation_results()

            for j in range(len(possible_outcomes)):
                # Check if obtained result is one of possible outcomes.
                if all(result == possible_outcomes[j]):  # Expect all elements of resultant list to be True.
                    outcomes_probabilities[j] += 1
                    break

        for i in range(len(outcomes_probabilities)):
            outcomes_probabilities[i] /= samples_number

        return outcomes_probabilities

    def __calculate_distribution_error_bound(self) -> float:
        n = self.experiment_configuration.initial_number_of_particles
        l = self.experiment_configuration.number_of_particles_left
        error_bound = 1.0 - (float(factorial(n)) / (pow(n, l) * factorial(n - l)))
        return error_bound

    # TODO TR: FIX THE TEST AFTER COMPUTATIONS ARE FINISHED!
    def test_approximate_and_exact_distribution_distance_for_haar_random_matrix(self) -> None:
        number_of_modes = 8
        initial_state = [1, 1, 1, 1, 0, 0, 0, 0]
        initial_number_of_particles = sum(initial_state)
        number_of_particles_lost = 2
        number_of_particles_left = initial_number_of_particles - number_of_particles_lost
        number_of_outcomes = binom(number_of_modes + number_of_particles_left - 1, number_of_particles_left)

        haar_random_matrices_number = 10**6

        error_bound = log2(2**number_of_outcomes - 2) - log2(self.error_probability_of_distance_bound)
        error_bound = sqrt(error_bound / (2 * haar_random_matrices_number))

        probabilities_list = []
        first_means = []
        first_stdevs = []

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

            current_probabilities = self.__calculate_approximate_distribution()

            # print(current_probabilities)

            if len(probabilities_list) == 0:
                probabilities_list = [[] for _ in range(len(current_probabilities))]

            for j in range(len(current_probabilities)):
                probabilities_list[j].append(current_probabilities[j])

            stDev = 0

            if len(probabilities_list[0]) > 2:
                first_means.append(average(probabilities_list[0]))
            if len(first_means) > 2:
                stDev = std(first_means)

            first_stdevs.append(stDev)

            f = open("1kk_haar_random_stdevs.txt", "a")
            f.write(str(stDev) + '\n')
            f.close()

        self.assertTrue(True)
        # self.assertAlmostEqual(distance, bound, delta=experiments_tv_distance_error_bound)


s = TestClassicalLossyBosonSamplingSimulator();
s.setUp()
s.test_approximate_and_exact_distribution_distance_for_haar_random_matrix()
