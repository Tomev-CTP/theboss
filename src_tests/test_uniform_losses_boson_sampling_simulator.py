__author__ = 'Tomasz Rybotycki'

import unittest
from math import factorial
from typing import List

from numpy import array, zeros
from scipy import special

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration, \
    BosonSamplingWithUniformLossesExactDistributionCalculator
from src.Quantum_Computations_Utilities import count_total_variation_distance, \
    count_tv_distance_error_bound_of_experiment_results
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

        number_of_samples = 1000
        number_of_outcomes = len(exact_distribution_calculator.get_outcomes_in_proper_order())
        error_probability_of_distance_bound = 0.001
        # Note, that this makes the test probabilistic in nature, but there's nothing one can do about that.
        error_bound_between_experiments_and_estimation = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes, samples_number=number_of_samples,
            error_probability=error_probability_of_distance_bound
        )

        approximate_distribution = self.__calculate_approximate_distribution(number_of_samples)
        distance_between_estimation_and_ideal = count_total_variation_distance(exact_distribution, approximate_distribution)
        error_bound_on_ideal_and_experiment = self.__calculate_uniform_loss_distribution_error_bound()
        self.assertLessEqual(distance_between_estimation_and_ideal, error_bound_on_ideal_and_experiment
                             + error_bound_between_experiments_and_estimation)

    def __calculate_uniform_loss_distribution_error_bound(self) -> float:
        """
            This is the distance bound between experimental and ideal results for uniform losses boson sampling.
            Basically this is capital Delta from [1]. Using eta, n and l notation from the paper [1] for readability
            purposes.
        :return: Distance bound between experimental and ideal results for uniform losses.
        """
        #
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
