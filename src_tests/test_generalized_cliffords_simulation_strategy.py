__author__ = 'Tomasz Rybotycki'

import unittest
from math import factorial
from typing import List

from numpy import array, average, log2, sqrt, zeros
from numpy.random import randint
from scipy import special
from scipy.special import binom

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration, \
    BosonSamplingWithFixedLossesExactDistributionCalculator
from src.Quantum_Computations_Utilities import count_total_variation_distance, \
    count_tv_distance_error_bound_of_experiment_results, generate_haar_random_unitary_matrix
from src.simulation_strategies.GeneralizedCliffordsSimulationStrategy import GeneralizedCliffordsSimulationStrategy


class TestGeneralizedCliffordsBosonSamplingSimulator(unittest.TestCase):

    def setUp(self) -> None:
        # Define variables for bound calculation.
        self.number_of_samples_for_distribution_approximation = 1000
        self.error_probability_of_distance_bound = 0.001

        # Generate permutation matrix and define initial state.
        self.permutation_matrix = array([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
        ])
        self.initial_state = [3, 1, 2, 1, 0]

        # Create configuration object.
        self.experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.permutation_matrix,
            initial_state=self.initial_state,
            initial_number_of_particles=sum(self.initial_state),
            number_of_modes=len(self.initial_state),
            number_of_particles_lost=0,
            number_of_particles_left=sum(self.initial_state),
            probability_of_uniform_loss=0.8
        )

        self.simulation_strategy = GeneralizedCliffordsSimulationStrategy(self.permutation_matrix)

        self.simulator = BosonSamplingSimulator(self.experiment_configuration.number_of_particles_left,
                                                self.experiment_configuration.initial_number_of_particles,
                                                self.experiment_configuration.number_of_modes,
                                                self.simulation_strategy)

    def test_approximate_and_exact_distribution_distance(self) -> None:
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()

        number_of_samples = 1000
        number_of_outcomes = len(exact_distribution_calculator.get_outcomes_in_proper_order())
        error_probability_of_distance_bound = 0.001
        # Note, that this makes the test probabilistic in nature, but there's nothing one can do about that.
        experiments_tv_distance_error_bound = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes, samples_number=number_of_samples,
            error_probability=error_probability_of_distance_bound
        )

        approximate_distribution = self.__calculate_approximate_distribution(number_of_samples)
        distance = count_total_variation_distance(exact_distribution, approximate_distribution)
        bound = self.__calculate_uniform_loss_distribution_error_bound()

        self.assertAlmostEqual(distance, bound, delta=experiments_tv_distance_error_bound)

    def __calculate_approximate_distribution(self, samples_number: int = 5000) -> List[float]:
        """
        Prepares the approximate distribution using boson sampling simulation method described by
        Oszmaniec and Brod. Obviously higher number of samples will generate better approximation.
        :return: Approximate distribution as a list.
        """
        exact_distribution_calculator = \
            BosonSamplingWithFixedLossesExactDistributionCalculator(self.experiment_configuration)

        possible_outcomes = exact_distribution_calculator.get_outcomes_in_proper_order()

        strategy = GeneralizedCliffordsSimulationStrategy(self.experiment_configuration.interferometer_matrix)

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

    def __calculate_uniform_loss_distribution_error_bound(self) -> float:
        # Using eta, n and l notation from the paper [1] for readability purposes.
        error_bound = 0
        n = self.experiment_configuration.initial_number_of_particles
        eta = self.experiment_configuration.probability_of_uniform_loss
        for number_of_particles_left in range(n + 1):
            l = number_of_particles_left
            subdistribution_weight = pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
            error_bound += subdistribution_weight * self.__calculate_fixed_loss_distribution_error_bound(n, l)
        return error_bound

    @staticmethod
    def __calculate_fixed_loss_distribution_error_bound(initial_number_of_particles: int,
                                                        number_of_particles_left: int) -> float:
        # Assigning n and l variables from [1] for more readable formula. This is Delta_{n, l} from the paper.
        n = initial_number_of_particles
        l = number_of_particles_left
        error_bound = 1.0 - (factorial(n) / (pow(n, l) * factorial(n - l)))
        return error_bound

    def test_approximate_and_exact_distribution_distance_for_haar_random_matrix(self) -> None:
        number_of_modes = 8
        initial_state = [1, 1, 1, 1, 0, 0, 0, 0]
        initial_number_of_particles = sum(initial_state)
        number_of_particles_lost = 0
        number_of_particles_left = initial_number_of_particles - number_of_particles_lost
        number_of_outcomes = binom(number_of_modes + number_of_particles_left - 1, number_of_particles_left)

        haar_random_matrices_number = 10 ** 2

        error_bound = log2(2 ** number_of_outcomes - 2) - log2(self.error_probability_of_distance_bound)
        error_bound = sqrt(error_bound / (2 * haar_random_matrices_number))

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

            current_probabilities = self.__calculate_approximate_distribution()

            if len(probabilities_list) == 0:
                probabilities_list = [[] for _ in range(len(current_probabilities))]

            for j in range(len(current_probabilities)):
                probabilities_list[j].append(current_probabilities[j])

        # Every probability should have probability 1 over outcomes number, if number of haar random matrices
        # goes to infinity. In that case I can select any probability.
        random_outcome_index = randint(0, len(current_probabilities))
        self.assertAlmostEqual(number_of_outcomes ** (-1), average(probabilities_list[random_outcome_index]),
                               delta=error_bound)
