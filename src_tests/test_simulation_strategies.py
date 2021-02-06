__author__ = "Tomasz Rybotycki"

"""
    The intention behind this code is to have tests for different simulation strategies in one place. The reason for
    that is that their code stays mostly the same and with every different strategy I end up writing more and more of
    the same stuff with only minor differences (mostly in experiments setup). 
"""

import unittest
from copy import deepcopy
from dataclasses import dataclass
from math import factorial
from typing import List

from numpy import array, average, ndarray
from numpy.random import randint
from scipy.special import binom

from src_tests import (
    BSDistributionCalculatorInterface,  BSSampleBasedDistributionCalculator, BSDistributionCalculatorWithFixedLosses,
    BSDistributionCalculatorWithUniformLosses,
    BosonSamplingExperimentConfiguration,
    SimulationStrategyFactory, StrategyType, SimulationStrategyInterface,
    LosslessNetworkSimulationStrategy,
    BSPermanentCalculatorFactory,
    generate_haar_random_unitary_matrix, calculate_number_of_possible_lossy_n_particle_m_mode_output_states,
    calculate_number_of_possible_n_particle_m_mode_output_states, count_total_variation_distance,
    count_tv_distance_error_bound_of_experiment_results
)


@dataclass
class DistributionAccuracyExperimentConfiguration:
    exact_calculator: BSDistributionCalculatorInterface
    estimation_calculator: BSSampleBasedDistributionCalculator
    approximation_distance_bound: float
    outcomes_number: int


class TestBosonSamplingClassicalSimulationStrategies(unittest.TestCase):

    def setUp(self) -> None:
        print(f"\nIn method {self._testMethodName}. Test start!\n")
        self._permutation_matrix = array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ], dtype=complex)

        self._number_of_samples_for_estimated_distribution_calculation = int(1e3)
        self._probability_of_error_in_distribution_calculation = 1e-3

        self._distance_calculation_initial_state = [1, 1, 1, 1, 0]
        self._distance_calculation_binned_initial_state = [2, 1, 1, 0, 0]
        self._distance_calculation_number_of_particles_lost = 2
        self._uniform_transmissivity = 0.8
        distance_calculation_initial_number_of_particles = sum(self._distance_calculation_initial_state)

        self._distributions_distance_experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._permutation_matrix,
            initial_state=array(self._distance_calculation_initial_state, dtype=int),
            initial_number_of_particles=distance_calculation_initial_number_of_particles,
            number_of_modes=len(self._distance_calculation_initial_state),
            number_of_particles_lost=self._distance_calculation_number_of_particles_lost,
            number_of_particles_left=distance_calculation_initial_number_of_particles -
                                     self._distance_calculation_number_of_particles_lost,
            uniform_transmissivity=0.8
        )

        permanent_calculator_factory = BSPermanentCalculatorFactory(self._permutation_matrix, None, None)
        self._bs_permanent_calculator = permanent_calculator_factory.generate_calculator()

        self._strategies_factory = SimulationStrategyFactory(self._distributions_distance_experiment_configuration,
                                                             self._bs_permanent_calculator)

        self._haar_random_matrices_number = 10
        self._haar_random_experiment_input_state = [1, 1, 1, 1, 0]
        haar_random_initial_number_of_particles = sum(self._haar_random_experiment_input_state)
        haar_random_number_of_particles_lost = 2
        self._haar_random_experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=array([], dtype=complex),
            initial_state=array(self._haar_random_experiment_input_state, dtype=int),
            initial_number_of_particles=haar_random_number_of_particles_lost,
            number_of_modes=len(self._haar_random_experiment_input_state),
            number_of_particles_lost=haar_random_number_of_particles_lost,
            number_of_particles_left=haar_random_initial_number_of_particles - haar_random_number_of_particles_lost
        )
        self._haar_random_binned_experiment_input_state = [2, 1, 1, 1, 0]

    def tearDown(self) -> None:
        print("\nTest finished!\n")

    def test_distribution_accuracy_for_fixed_losses_strategy(self) -> None:
        self.__prepare_lossy_distance_experiment_settings()
        self._distributions_distance_experiment_configuration.network_simulation_strategy = \
            LosslessNetworkSimulationStrategy(self._permutation_matrix)

        self._strategies_factory.experiment_configuration = self._distributions_distance_experiment_configuration
        self._strategies_factory.strategy_type = StrategyType.FIXED_LOSS

        distance_experiment_configuration = DistributionAccuracyExperimentConfiguration(
            exact_calculator=BSDistributionCalculatorWithFixedLosses(
                self._distributions_distance_experiment_configuration,
                self._bs_permanent_calculator),
            estimation_calculator=self.__generate_estimated_distribution_calculator(
                self._strategies_factory.generate_strategy()
            ),
            outcomes_number=calculate_number_of_possible_n_particle_m_mode_output_states(
                n=self._distributions_distance_experiment_configuration.number_of_particles_left,
                m=self._distributions_distance_experiment_configuration.number_of_modes
            ),
            approximation_distance_bound=self.__calculate_fixed_losses_distance_bound_from_exact_to_experimental(
                self._distributions_distance_experiment_configuration.initial_number_of_particles,
                self._distributions_distance_experiment_configuration.number_of_particles_left
            )
        )
        self.__perform_distance_of_approximate_distribution_from_ideal(distance_experiment_configuration)

    def __prepare_lossy_distance_experiment_settings(self) -> None:
        # At least for now lossy experiments are only for classic [1, 1, ..., 1, 0, 0, ..., 0] states.
        self._distributions_distance_experiment_configuration.initial_state = self._distance_calculation_initial_state
        initial_number_of_particles = sum(self._distance_calculation_initial_state)
        self._distributions_distance_experiment_configuration.initial_number_of_particles = initial_number_of_particles
        self._distributions_distance_experiment_configuration.number_of_particles_lost = \
            self._distance_calculation_number_of_particles_lost
        self._distributions_distance_experiment_configuration.number_of_particles_left = \
            initial_number_of_particles - self._distance_calculation_number_of_particles_lost
        self._distributions_distance_experiment_configuration.uniform_transmissivity = self._uniform_transmissivity

    def __perform_distance_of_approximate_distribution_from_ideal(
            self, distance_accuracy_experiment_configuration: DistributionAccuracyExperimentConfiguration) -> None:
        distance_from_exact_to_estimated = self.__calculate_distance_from_exact_distribution_to_estimated_distribution(
            exact_distribution_calculator=distance_accuracy_experiment_configuration.exact_calculator,
            estimated_distribution_calculator=distance_accuracy_experiment_configuration.estimation_calculator
        )

        distance_from_approximation_to_estimated = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=distance_accuracy_experiment_configuration.outcomes_number,
            samples_number=self._number_of_samples_for_estimated_distribution_calculation,
            error_probability=self._probability_of_error_in_distribution_calculation
        )

        # Using triangle inequality of (TV) distance.
        max_allowed_distance = \
            distance_accuracy_experiment_configuration.approximation_distance_bound \
            + distance_from_approximation_to_estimated

        self.assertLessEqual(distance_from_exact_to_estimated, max_allowed_distance,
                             f"Distance from exact distribution ({distance_from_exact_to_estimated}) is "
                             f"greater than maximum distance allowed ({max_allowed_distance}).")

    def __generate_estimated_distribution_calculator(self, strategy: SimulationStrategyInterface,
                                                     outcomes: List[ndarray] = None) \
            -> BSSampleBasedDistributionCalculator:
        estimated_distribution_calculator = BSSampleBasedDistributionCalculator(
            experiment_configuration=self._distributions_distance_experiment_configuration,
            strategy=strategy,
            outcomes=outcomes
        )
        return estimated_distribution_calculator

    def __calculate_distance_from_exact_distribution_to_estimated_distribution(
            self, exact_distribution_calculator: BSDistributionCalculatorInterface,
            estimated_distribution_calculator: BSSampleBasedDistributionCalculator) -> float:
        """
            Using specified calculators, it counts exact and estimated probability distributions and calculates
            the tv distance between them.
            :param exact_distribution_calculator: Calculator of the exact (ideal, bruteforce permanents) distribution.
            :param estimated_distribution_calculator: Calculator of the approximate (sampling according to strategy)
            distribution.
            :return: TV distance between exact and approximated distribution.
        """
        exact_distribution = exact_distribution_calculator.calculate_distribution()
        approximated_distribution = estimated_distribution_calculator.calculate_approximate_distribution(
            samples_number=self._number_of_samples_for_estimated_distribution_calculation
        )
        return count_total_variation_distance(exact_distribution, approximated_distribution)

    @staticmethod
    def __calculate_fixed_losses_distance_bound_from_exact_to_experimental(initial_number_of_particles: int,
                                                                           number_of_particles_left: int) -> float:
        """
            This is the distance bound between experimental and ideal results for fixed losses boson sampling.
            Basically this is capital Delta from [1]. Using n and l notation from the paper [1] for readability
            purposes.
            This will be used for uniform losses and fixed losses bound calculation.
            :return: Distance bound between experimental and ideal results for uniform losses.
        """
        n = initial_number_of_particles
        l = number_of_particles_left
        error_bound = 1.0 - (factorial(n) / (pow(n, l) * factorial(n - l)))
        return error_bound

    def test_distribution_accuracy_for_generalized_cliffords_strategy(self) -> None:
        self.__prepare_lossless_distance_experiment_settings()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests()

    def __prepare_lossless_distance_experiment_settings(self) -> None:
        self._distributions_distance_experiment_configuration.initial_state = self._distance_calculation_initial_state
        initial_number_of_particles = sum(self._distance_calculation_initial_state)
        self._distributions_distance_experiment_configuration.initial_number_of_particles = initial_number_of_particles
        self._distributions_distance_experiment_configuration.number_of_particles_lost = 0
        self._distributions_distance_experiment_configuration.number_of_particles_left = initial_number_of_particles
        self._distributions_distance_experiment_configuration.uniform_transmissivity = 1

    def __continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(self,
                                                                                           strategy: StrategyType = StrategyType.GENERALIZED_CLIFFORD) -> None:
        self._strategies_factory.experiment_configuration = self._distributions_distance_experiment_configuration
        self._strategies_factory.strategy_type = strategy
        distance_experiment_configuration = DistributionAccuracyExperimentConfiguration(
            # This exact calculator, when there are no losses, will do the work just fine.
            exact_calculator=BSDistributionCalculatorWithFixedLosses(
                self._distributions_distance_experiment_configuration,
                self._bs_permanent_calculator),
            estimation_calculator=self.__generate_estimated_distribution_calculator(
                self._strategies_factory.generate_strategy()
            ),
            outcomes_number=calculate_number_of_possible_n_particle_m_mode_output_states(
                n=self._distributions_distance_experiment_configuration.number_of_particles_left,
                m=self._distributions_distance_experiment_configuration.number_of_modes
            ),
            approximation_distance_bound=0  # This strategy returns exact solution.
        )
        self.__perform_distance_of_approximate_distribution_from_ideal(distance_experiment_configuration)

    def test_distribution_accuracy_for_generalized_cliffords_strategy_with_binned_input(self) -> None:
        self.__prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests()

    def __prepare_lossless_distance_experiments_settings_with_binned_inputs(self) -> None:
        self._distributions_distance_experiment_configuration.initial_state = \
            self._distance_calculation_binned_initial_state
        initial_number_of_particles = sum(self._distance_calculation_binned_initial_state)
        self._distributions_distance_experiment_configuration.initial_number_of_particles = initial_number_of_particles
        self._distributions_distance_experiment_configuration.number_of_particles_lost = 0
        self._distributions_distance_experiment_configuration.number_of_particles_left = initial_number_of_particles
        self._distributions_distance_experiment_configuration.uniform_transmissivity = 1

    def test_distribution_accuracy_for_lossy_network_generalized_cliffords_strategy_without_losses(self):
        self.__prepare_lossless_distance_experiment_settings()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(
            StrategyType.LOSSY_NET_GENERALIZED_CLIFFORD)

    def test_distribution_accuracy_for_lossy_network_generalized_cliffords_strategy_without_losses_with_binned_input(
            self):
        self.__prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(
            StrategyType.LOSSY_NET_GENERALIZED_CLIFFORD)

    def test_distribution_accuracy_for_uniform_losses_generalized_cliffords_without_losses(self):
        self.__prepare_lossless_distance_experiment_settings()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(
            StrategyType.GENERALIZED_U_LOSSY_CLIFFORD)

    def test_distribution_accuracy_for_uniform_losses_generalized_cliffords_without_losses_with_binned_input(self):
        self.__prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(
            StrategyType.GENERALIZED_U_LOSSY_CLIFFORD)

    def test_distribution_accuracy_for_nonuniform_losses_approximated_strategy_without_approximated_modes_or_loses(self):
        self.__prepare_lossless_distance_experiment_settings()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(
            StrategyType.NONUNIFORM_APPROXIMATED)

    def test_distribution_accuracy_for_nonuniform_losses_approximated_strategy_without_approximated_modes_or_loses_with_binned_input(self):
        self.__prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self.__continue_with_common_part_of_lossless_generalized_cliffords_based_strategy_tests(
            StrategyType.NONUNIFORM_APPROXIMATED)

    def test_distribution_accuracy_for_uniform_losses_strategy(self) -> None:
        self.__prepare_lossy_distance_experiment_settings()
        self._strategies_factory.experiment_configuration = self._distributions_distance_experiment_configuration
        self._strategies_factory.strategy_type = StrategyType.UNIFORM_LOSS
        exact_calculator = BSDistributionCalculatorWithUniformLosses(
            self._distributions_distance_experiment_configuration, self._bs_permanent_calculator)
        distance_experiment_configuration = DistributionAccuracyExperimentConfiguration(
            exact_calculator=exact_calculator,
            estimation_calculator=self.__generate_estimated_distribution_calculator(
                self._strategies_factory.generate_strategy(),
                outcomes=exact_calculator.get_outcomes_in_proper_order()
            ),
            outcomes_number=calculate_number_of_possible_lossy_n_particle_m_mode_output_states(
                n=self._distributions_distance_experiment_configuration.number_of_particles_left,
                m=self._distributions_distance_experiment_configuration.number_of_modes
            ),
            approximation_distance_bound=self.__calculate_uniform_loss_distribution_error_bound(),
        )
        self.__perform_distance_of_approximate_distribution_from_ideal(distance_experiment_configuration)

    def __calculate_uniform_loss_distribution_error_bound(self) -> float:
        """
            This is the distance bound between approximated and ideal results for uniform losses boson sampling.
            Basically this is capital Delta from [1]. Using eta, n and l notation from the paper [1] for readability
            purposes.
        :return: Distance bound between experimental and ideal results for uniform losses.
        """
        #
        error_bound = 0
        n = self._distributions_distance_experiment_configuration.initial_number_of_particles
        eta = self._distributions_distance_experiment_configuration.uniform_transmissivity
        for number_of_particles_left in range(n + 1):
            l = number_of_particles_left
            subdistribution_weight = pow(eta, l) * binom(n, l) * pow(1.0 - eta, n - l)
            error_bound += \
                subdistribution_weight * self.__calculate_fixed_losses_distance_bound_from_exact_to_experimental(n, l)
        return error_bound

    def test_distribution_accuracy_for_u_lossy_generalized_cliffords_strategy_with_losses(self):
        self.__prepare_lossy_distance_experiment_settings()
        self.__continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(StrategyType.GENERALIZED_U_LOSSY_CLIFFORD)

    def __continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(self,
                                                                                                  strategy: StrategyType = StrategyType.GENERALIZED_U_LOSSY_CLIFFORD) -> None:
        self._strategies_factory.experiment_configuration = self._distributions_distance_experiment_configuration
        self._strategies_factory.strategy_type = strategy
        exact_calculator = BSDistributionCalculatorWithUniformLosses(
            self._distributions_distance_experiment_configuration, self._bs_permanent_calculator)

        if strategy == StrategyType.LOSSY_NET_GENERALIZED_CLIFFORD or strategy == StrategyType.NONUNIFORM_APPROXIMATED:
            self._strategies_factory.bs_permanent_calculator.matrix *= pow(self._uniform_transmissivity, 0.5)

        distance_experiment_configuration = DistributionAccuracyExperimentConfiguration(
            # This exact calculator, when there are no losses, will do the work just fine.
            exact_calculator=exact_calculator,
            estimation_calculator=self.__generate_estimated_distribution_calculator(
                self._strategies_factory.generate_strategy(),
                outcomes=exact_calculator.get_outcomes_in_proper_order()
            ),
            outcomes_number=calculate_number_of_possible_lossy_n_particle_m_mode_output_states(
                n=self._distributions_distance_experiment_configuration.number_of_particles_left,
                m=self._distributions_distance_experiment_configuration.number_of_modes
            ),
            approximation_distance_bound=0  # This strategy returns exact solution.
        )
        self.__perform_distance_of_approximate_distribution_from_ideal(distance_experiment_configuration)

    def test_distribution_accuracy_for_net_lossy_generalized_cliffords_strategy_with_losses(self):
        self.__prepare_lossy_distance_experiment_settings()
        self.__continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(
            StrategyType.LOSSY_NET_GENERALIZED_CLIFFORD)

    def test_distribution_accuracy_for_u_lossy_generalized_cliffords_strategy_with_losses_and_binned_input(self):
        self.__prepare_lossy_distance_experiment_settings_with_binned_input()
        self.__continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(
            StrategyType.GENERALIZED_U_LOSSY_CLIFFORD)

    def __prepare_lossy_distance_experiment_settings_with_binned_input(self):
        # At least for now lossy experiments are only for classic [1, 1, ..., 1, 0, 0, ..., 0] states.
        self._distributions_distance_experiment_configuration.initial_state = self._distance_calculation_binned_initial_state
        initial_number_of_particles = sum(self._distance_calculation_initial_state)
        self._distributions_distance_experiment_configuration.initial_number_of_particles = initial_number_of_particles
        self._distributions_distance_experiment_configuration.number_of_particles_lost = \
            self._distance_calculation_number_of_particles_lost
        self._distributions_distance_experiment_configuration.number_of_particles_left = \
            initial_number_of_particles - self._distance_calculation_number_of_particles_lost
        self._distributions_distance_experiment_configuration.uniform_transmissivity = self._uniform_transmissivity

    def test_distribution_accuracy_for_net_lossy_generalized_cliffords_strategy_with_losses_and_binned_input(self):
        self.__prepare_lossy_distance_experiment_settings_with_binned_input()
        self.__continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(
            StrategyType.LOSSY_NET_GENERALIZED_CLIFFORD)

    def test_distribution_accuracy_for_nonuniform_losses_approximated_strategy_without_approximated_modes(self):
        self.__prepare_lossy_distance_experiment_settings()
        self.__continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(StrategyType.NONUNIFORM_APPROXIMATED)

    def test_distribution_accuracy_for_nonuniform_losses_approximated_strategy_without_approximated_modes_with_binned_input(self):
        self.__prepare_lossy_distance_experiment_settings_with_binned_input()
        self.__continue_with_common_part_of_uniformly_lossy_generalized_cliffords_based_strategy_tests(StrategyType.NONUNIFORM_APPROXIMATED)

    def test_haar_random_interferometers_distance_for_fixed_losses_strategy(self) -> None:
        self.__set_experiment_configuration_for_standard_haar_random()
        strategy_factory = SimulationStrategyFactory(self._haar_random_experiment_configuration,
                                                     self._bs_permanent_calculator,
                                                     StrategyType.FIXED_LOSS)
        self.__test_haar_random_interferometers_approximation_distance_from_ideal(strategy_factory)

    def test_haar_random_interferometers_distance_for_cliffords_r_strategy(self) -> None:
        self.__set_experiment_configuration_for_lossless_haar_random()
        strategy_factory = SimulationStrategyFactory(self._haar_random_experiment_configuration,
                                                     self._bs_permanent_calculator,
                                                     StrategyType.CLIFFORD_R)
        self.__test_haar_random_interferometers_approximation_distance_from_ideal(strategy_factory)

    def test_haar_random_interferometers_distance_for_generalized_cliffords_strategy(self) -> None:
        self.__set_experiment_configuration_for_lossless_haar_random()
        strategy_factory = SimulationStrategyFactory(self._haar_random_experiment_configuration,
                                                     self._bs_permanent_calculator,
                                                     StrategyType.GENERALIZED_CLIFFORD)
        self.__test_haar_random_interferometers_approximation_distance_from_ideal(strategy_factory)

    def test_haar_random_interferometers_distance_for_generalized_cliffords_strategy_with_binned_input(self) -> None:
        self.__set_experiment_configuration_for_binned_haar_random()
        strategy_factory = SimulationStrategyFactory(self._haar_random_experiment_configuration,
                                                     self._bs_permanent_calculator,
                                                     StrategyType.GENERALIZED_CLIFFORD)
        self.__test_haar_random_interferometers_approximation_distance_from_ideal(strategy_factory)

    def __set_experiment_configuration_for_binned_haar_random(self) -> None:
        # Note that binned configuration is also for lossless case (for now)
        self._haar_random_experiment_configuration.initial_state = self._haar_random_binned_experiment_input_state
        number_of_particles_in_the_experiment = len(self._haar_random_binned_experiment_input_state)
        self._haar_random_experiment_configuration.initial_number_of_particles = number_of_particles_in_the_experiment
        self._haar_random_experiment_configuration.number_of_particles_left = number_of_particles_in_the_experiment

    def __set_experiment_configuration_for_standard_haar_random(self) -> None:
        self._haar_random_experiment_configuration.initial_state = self._haar_random_experiment_input_state
        number_of_particles_in_the_experiment = sum(self._haar_random_experiment_input_state)
        self._haar_random_experiment_configuration.initial_number_of_particles = number_of_particles_in_the_experiment
        self._haar_random_experiment_configuration.number_of_particles_left = \
            number_of_particles_in_the_experiment - self._haar_random_experiment_configuration.number_of_particles_lost

    def __set_experiment_configuration_for_lossless_haar_random(self) -> None:
        self._haar_random_experiment_configuration.initial_state = self._haar_random_experiment_input_state
        number_of_particles_in_the_experiment = sum(self._haar_random_experiment_input_state)
        self._haar_random_experiment_configuration.initial_number_of_particles = number_of_particles_in_the_experiment
        self._haar_random_experiment_configuration.number_of_particles_left = number_of_particles_in_the_experiment

    def __test_haar_random_interferometers_approximation_distance_from_ideal(
            self, strategy_factory: SimulationStrategyFactory) -> None:
        number_of_outcomes = calculate_number_of_possible_n_particle_m_mode_output_states(
            n=self._haar_random_experiment_configuration.number_of_particles_left,
            m=self._haar_random_experiment_configuration.number_of_modes
        )

        error_bound = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes, samples_number=self._haar_random_matrices_number,
            error_probability=self._probability_of_error_in_distribution_calculation
        )

        probabilities_list = []
        current_probabilities = []

        for i in range(self._haar_random_matrices_number):

            print(f"Current Haar random matrix index: {i} out of {self._haar_random_matrices_number}.")

            experiment_configuration = deepcopy(self._haar_random_experiment_configuration)
            experiment_configuration.interferometer_matrix = generate_haar_random_unitary_matrix(
                self._haar_random_experiment_configuration.number_of_modes)
            strategy_factory.experiment_configuration = experiment_configuration
            self._bs_permanent_calculator.matrix = experiment_configuration.interferometer_matrix
            strategy_factory.bs_permanent_calculator = self._bs_permanent_calculator
            distribution_calculator = BSSampleBasedDistributionCalculator(experiment_configuration,
                                                                          strategy_factory.generate_strategy())

            current_probabilities = distribution_calculator.calculate_approximate_distribution(
                samples_number=self._number_of_samples_for_estimated_distribution_calculation)

            if len(probabilities_list) == 0:
                probabilities_list = [[] for _ in range(len(current_probabilities))]

            for j in range(len(current_probabilities)):
                probabilities_list[j].append(current_probabilities[j])

        random_outcome_index = randint(0, len(current_probabilities))
        self.assertAlmostEqual(number_of_outcomes ** (-1), average(probabilities_list[random_outcome_index]),
                               delta=error_bound)
