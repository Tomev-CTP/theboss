__author__ = 'Tomasz Rybotycki'

import unittest
from math import factorial
from typing import List

from numpy import array, zeros
from scipy import special

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration, \
    BosonSamplingWithUniformLossesExactDistributionCalculator
from src.Quantum_Computations_Utilities import calculate_total_variation_distance
from src.simulation_strategies.UniformLossSimulationStrategy import UniformLossSimulationStrategy


class TestClassicalLossyBosonSamplingSimulator(unittest.TestCase):

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
        self.initial_state = [1, 1, 1, 1, 0]

        # Create configuration object.
        self.experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self.permutation_matrix,
            initial_state=self.initial_state,
            initial_number_of_particles=sum(self.initial_state),
            number_of_modes=len(self.initial_state),
            number_of_particles_lost=self.number_of_particles_lost,
            number_of_particles_left=sum(self.initial_state) - self.number_of_particles_lost,
            probability_of_uniform_loss=0.8
        )

        self.simulation_strategy = \
            UniformLossSimulationStrategy(self.permutation_matrix,
                                          self.experiment_configuration.number_of_modes,
                                          self.experiment_configuration.probability_of_uniform_loss)

        self.simulator = BosonSamplingSimulator(self.experiment_configuration.number_of_particles_left,
                                                self.experiment_configuration.initial_number_of_particles,
                                                self.experiment_configuration.number_of_modes,
                                                self.simulation_strategy)

    def test_approximate_and_exact_distribution_distance(self) -> None:
        exact_distribution_calculator = \
            BosonSamplingWithUniformLossesExactDistributionCalculator(self.experiment_configuration)
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()
        # TR TODO: Delta in assertion should depend on samples number!
        approximate_distribution = self.__calculate_approximate_distribution()
        distance = calculate_total_variation_distance(exact_distribution, approximate_distribution)
        bound = self.__calculate_uniform_loss_distribution_error_bound()
        self.assertAlmostEqual(distance, bound, delta=1e-2)

    def __calculate_approximate_distribution(self, samples_number: int = 10000) -> List[float]:
        """
        Prepares the approximate distribution using boson sampling simulation method described by
        Oszmaniec and Brod. Obviously higher number of samples will generate better approximation.
        :return: Approximate distribution as a list.
        """
        exact_distribution_calculator = \
            BosonSamplingWithUniformLossesExactDistributionCalculator(self.experiment_configuration)

        possible_outcomes = exact_distribution_calculator.get_outcomes_in_proper_order()

        outcomes_probabilities = zeros(len(possible_outcomes))

        for i in range(samples_number):
            result = self.simulator.get_classical_simulation_results()

            for j in range(len(possible_outcomes)):
                # Check if obtained result is one of possible outcomes.
                if not (result == possible_outcomes[j]).__contains__(False):
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
