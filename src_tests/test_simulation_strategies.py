__author__ = 'Tomasz Rybotycki'

"""
The intention behind this code is to have tests for different simulation strategies in one place. The reason for that
is that their code stays mostly the same and with every different strategy I end up writing more and more of the same
stuff with only minor differences (mostly in experiments setup). 
"""

import unittest
from math import factorial
from numpy import array, ndarray, asarray
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration
from src.Quantum_Computations_Utilities import count_tv_distance_error_bound_of_experiment_results, \
    count_total_variation_distance
from src.Boson_Sampling_Utilities import calculate_number_of_possible_n_particle_m_mode_output_states
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingWithFixedLossesExactDistributionCalculator, \
    BosonSamplingWithUniformLossesExactDistributionCalculator, BosonSamplingExactDistributionCalculator
from src_tests.common_code_for_tests import ApproximateDistributionCalculator


class TestBosonSamplingClassicalSimulationStrategies(unittest.TestCase):

    def setUp(self) -> None:
        self._permutation_matrix = array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ])

        self._number_of_samples_for_estimated_distribution_calculation = int(1e3)
        self._probability_of_error_in_distribution_calculation = 1e-3

        distance_calculation_initial_state = [1, 1, 1, 1, 0]
        distance_calculation_number_of_particles_lost = 2
        distance_calculation_initial_number_of_particles = sum(distance_calculation_initial_state)

        self._fixed_losses_experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._permutation_matrix,
            initial_state=asarray(distance_calculation_initial_state),
            initial_number_of_particles=distance_calculation_initial_number_of_particles,
            number_of_modes=len(distance_calculation_initial_state),
            number_of_particles_lost=distance_calculation_number_of_particles_lost,
            number_of_particles_left=distance_calculation_initial_number_of_particles - distance_calculation_number_of_particles_lost
        )

        self._haar_random_matrices_number = 100
        haar_random_experiment_input_state = [1, 1, 1, 1, 0, 0, 0, 0]
        haar_random_initial_number_of_particles = sum(haar_random_experiment_input_state)
        haar_random_number_of_particles_lost = 2
        self._haar_random_experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=[],
            initial_state=asarray(haar_random_experiment_input_state),
            initial_number_of_particles=haar_random_number_of_particles_lost,
            number_of_modes=len(haar_random_initial_number_of_particles),
            number_of_particles_lost=haar_random_number_of_particles_lost,
            number_of_particles_left=haar_random_experiment_input_state - haar_random_number_of_particles_lost
        )

    def test_distribution_accuracy_for_fixed_losses_strategy(self) -> None:
        # Using triangle inequality of (TV) distance.
        exact_distribution_calculator = BosonSamplingWithFixedLossesExactDistributionCalculator(
            self._fixed_losses_experiment_configuration
        )

        approximate_distribution_calculator = self.__generate_fixed_losses_approximate_distribution_calculator()

        distance_from_exact_to_estimated = self.__calculate_distance_from_exact_distribution_to_estimated_distribution(
            exact_distribution_calculator=exact_distribution_calculator,
            approximate_distribution_calculator=approximate_distribution_calculator
        )

        distance_from_exact_to_experimental = self.__calculate_fixed_losses_distance_bound_from_exact_to_experimental()

        distance_from_experimental_to_estimated = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=calculate_number_of_possible_n_particle_m_mode_output_states(
                n=self._fixed_losses_experiment_configuration.number_of_particles_left,
                m=self._fixed_losses_experiment_configuration.number_of_modes
            ),
            samples_number=self._number_of_samples_for_estimated_distribution_calculation,
            error_probability=self._probability_of_error_in_distribution_calculation
        )

        max_allowed_distance = distance_from_exact_to_experimental + distance_from_experimental_to_estimated

        self.__assert_distance_is_within_bounds(distance_from_exact_to_estimated, max_allowed_distance)

    def __generate_fixed_losses_approximate_distribution_calculator(self) -> ApproximateDistributionCalculator:
        """
            Prepares an approximate distribution calculator for fixed losses regime.
        """
        strategy_for_approximate_simulator = \
            FixedLossSimulationStrategy(self._fixed_losses_experiment_configuration.interferometer_matrix,
                                        self._fixed_losses_experiment_configuration.number_of_particles_left,
                                        self._fixed_losses_experiment_configuration.number_of_modes)

        approximate_distribution_calculator = ApproximateDistributionCalculator(
            experiment_configuration=self._fixed_losses_experiment_configuration,
            strategy=strategy_for_approximate_simulator
        )

        return approximate_distribution_calculator

    def __calculate_distance_from_exact_distribution_to_estimated_distribution(self,
                                                                               exact_distribution_calculator: BosonSamplingExactDistributionCalculator,
                                                                               approximate_distribution_calculator: ApproximateDistributionCalculator) -> float:
        """
            Using specified calculators, it counts exact and estimated probability distributions and calculates
            the tv distance between them.
            :param exact_distribution_calculator: Calculator of the exact (ideal, bruteforce permanents) distribution.
            :param approximate_distribution_calculator: Calculator of the approximate (sampling according to strategy)
            distribution.
            :return: TV distance between exact and approximated distribution.
        """
        exact_distribution = exact_distribution_calculator.calculate_exact_distribution()
        approximated_distribution = approximate_distribution_calculator.calculate_approximate_distribution(
            samples_number=self._number_of_samples_for_estimated_distribution_calculation
        )
        return count_total_variation_distance(exact_distribution, approximated_distribution)

    def __calculate_fixed_losses_distance_bound_from_exact_to_experimental(self) -> float:
        return self.__calculate_distance_bound_from_exact_to_experimental(
            initial_number_of_particles=self._fixed_losses_experiment_configuration.initial_number_of_particles,
            number_of_particles_lost=self._fixed_losses_experiment_configuration.number_of_particles_lost
        )

    @staticmethod
    def __calculate_distance_bound_from_exact_to_experimental(initial_number_of_particles: int,
                                                              number_of_particles_lost: int):
        """
            This is the distance bound between experimental and ideal results for uniform losses boson sampling.
            Basically this is capital Delta from [1]. Using n and l notation from the paper [1] for readability
            purposes.
            This will be used for uniform losses and fixed losses bound calculation.
            :return: Distance bound between experimental and ideal results for uniform losses.
        """
        n = initial_number_of_particles
        l = initial_number_of_particles - number_of_particles_lost
        error_bound = 1.0 - (factorial(n) / (pow(n, l) * factorial(n - l)))
        return error_bound

    def __assert_distance_is_within_bounds(self, distance: float, max_allowed_distance: float) -> None:
        """ This is common for many strategies, therefore should be separated."""
        self.assertLessEqual(distance, max_allowed_distance, f'Distance from exact distribution ({distance}) is '
                                                             f'greater than maximum distance allowed ({max_allowed_distance}).')

    def test_approximate_and_exact_distribution_distance_for_haar_random_matrix(self) -> None:
        number_of_modes = 8
        initial_state = [1, 1, 1, 1, 0, 0, 0, 0]
        initial_number_of_particles = sum(initial_state)
        number_of_particles_lost = 2
        number_of_particles_left = initial_number_of_particles - number_of_particles_lost
        number_of_outcomes = calculate_number_of_possible_n_particle_m_mode_output_states(number_of_particles_left,
                                                                                          number_of_modes)

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
