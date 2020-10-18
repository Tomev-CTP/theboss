import unittest
from random import uniform
from typing import List

from numpy import asarray, ndarray

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.LossyBosonSamplingExactDistributionCalculators import (
    BosonSamplingExperimentConfiguration, BosonSamplingWithUniformLossesExactDistributionCalculator)
from src.network_simulation_strategy.LossyNetworkSimulationStrategy import LossyNetworkSimulationStrategy
from src.Quantum_Computations_Utilities import count_total_variation_distance, \
    count_tv_distance_error_bound_of_experiment_results, generate_haar_random_unitary_matrix
from src.SimulationStrategyFactory import SimulationStrategyFactory, StrategyTypes
from src_tests.common_code_for_tests import ApproximateDistributionCalculator


class TestBosonSamplingClassicalSimulationStrategies(unittest.TestCase):

    def setUp(self) -> None:
        self._probability_of_uniform_loss = 0.8
        self._initial_state = [1, 1, 1, 1, 0, 0, 0, 0]
        self._number_of_samples_for_experiments = 1000
        self._probability_of_error_in_distribution_calculation = 0.001
        self._lossy_interferometer_matrix = self._probability_of_uniform_loss * generate_haar_random_unitary_matrix(
            len(self._initial_state))
        self._experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._lossy_interferometer_matrix,
            initial_state=asarray(self._initial_state),
            initial_number_of_particles=sum(self._initial_state),
            number_of_modes=len(self._initial_state),
            number_of_particles_lost=0,  # Losses should only come from network.
            number_of_particles_left=sum(self._initial_state),
            probability_of_uniform_loss=self._probability_of_uniform_loss,
            network_simulation_strategy=LossyNetworkSimulationStrategy(self._lossy_interferometer_matrix)
        )
        self._strategy_factory = SimulationStrategyFactory(self._experiment_configuration)

    def test_lossy_network_simulation_number_of_particles(self) -> None:
        """
        This test checks if number of particles for lossy network simulator is lower than number of
        particles without the losses. Note that I can be pretty sure it will be lower in large losses
        regime, but in case of lower losses this test may not hold.
        """
        self._strategy_factory.set_strategy_type(StrategyTypes.FIXED_LOSS)

        strategy = self._strategy_factory.generate_strategy()
        simulator = BosonSamplingSimulator(strategy)
        lossy_average_number_of_particles = 0

        for _ in range(self._number_of_samples_for_experiments):
            lossy_average_number_of_particles += sum(simulator.get_classical_simulation_results(self._initial_state))

        lossy_average_number_of_particles /= self._number_of_samples_for_experiments
        lossless_number_of_particles = sum(self._initial_state)
        self.assertLess(lossy_average_number_of_particles, lossless_number_of_particles)

    def test_lossy_network_with_uniform_losses_distribution_accuracy(self) -> None:
        """
        This test is for uniform losses. It checks if uniform losses in network simulations are close to
        simulation with uniform losses at the input. Given that both approximate the same quantity only
        sampling complexity error is assumed.
        :return:
        """
        calculator = BosonSamplingWithUniformLossesExactDistributionCalculator(self._experiment_configuration)
        outcomes = calculator.get_outcomes_in_proper_order()
        outcomes_number = len(outcomes)

        distance_bound_between_estimated_distributions = 2 * count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=outcomes_number,
            samples_number=self._number_of_samples_for_experiments,
            error_probability=self._probability_of_error_in_distribution_calculation
        )

        estimated_distribution_calculator = ApproximateDistributionCalculator(
            experiment_configuration=self._experiment_configuration,
            strategy=self._strategy_factory.generate_strategy(),
            outcomes=outcomes
        )

        lossy_network_distribution = estimated_distribution_calculator.calculate_approximate_distribution()

        self._strategy_factory.set_strategy_type(StrategyTypes.UNIFORM_LOSS)

        estimated_distribution_calculator = ApproximateDistributionCalculator(
            experiment_configuration=self._experiment_configuration,
            strategy=self._strategy_factory.generate_strategy(),
            outcomes=outcomes
        )

        uniform_losses_distribution = estimated_distribution_calculator.calculate_approximate_distribution()

        distance_between_distributions = count_total_variation_distance(
            lossy_network_distribution, uniform_losses_distribution
        )

        self.assertLessEqual(distance_between_distributions, distance_bound_between_estimated_distributions)

    def test_lossy_network_with_uniform_losses_distribution_accuracy_against_generalized_cliffords(self) -> None:
        """
        This test checks if lossy networks distribution

        """
        calculator = BosonSamplingWithUniformLossesExactDistributionCalculator(self._experiment_configuration)
        outcomes = calculator.get_outcomes_in_proper_order()
        outcomes_number = len(outcomes)

        distance_bound_between_estimated_distributions = 2 * count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=outcomes_number,
            samples_number=self._number_of_samples_for_experiments,
            error_probability=self._probability_of_error_in_distribution_calculation
        )

        estimated_distribution_calculator = ApproximateDistributionCalculator(
            experiment_configuration=self._experiment_configuration,
            strategy=self._strategy_factory.generate_strategy(),
            outcomes=outcomes
        )

        lossy_network_distribution = estimated_distribution_calculator.calculate_approximate_distribution()
        generalized_cliffords_distribution = self.__calculate_generalized_cliffords_distribution_with_lossy_inputs(
            outcomes)

        distance_between_distributions = count_total_variation_distance(
            lossy_network_distribution, generalized_cliffords_distribution
        )

        self.assertLessEqual(distance_between_distributions, distance_bound_between_estimated_distributions)

    def __calculate_generalized_cliffords_distribution_with_lossy_inputs(self, outcomes: List[List[int]]) -> List[
        float]:
        """
        This method calculates approximate distribution for lossy states using generalized cliffords
        method.
        :param outcomes: Possible outcomes list.
        :return: Approximate distribution.
        """
        probabilities = [0] * len(outcomes)
        self._strategy_factory.set_strategy_type(StrategyTypes.GENERALIZED_CLIFFORD)
        strategy = self._strategy_factory.generate_strategy()
        simulator = BosonSamplingSimulator(strategy)

        for i in range(self._number_of_samples_for_experiments):

            result = simulator.get_classical_simulation_results(self.__get_uniformly_lossy_input_state())

            for j in range(len(outcomes)):
                # Check if obtained result is one of possible outcomes.
                if all(result == outcomes[j]):  # Expect all elements of resultant list to be True.
                    probabilities[j] += 1
                    break

        for i in range(len(probabilities)):
            probabilities[i] /= self._number_of_samples_for_experiments

        return probabilities

    def __get_uniformly_lossy_input_state(self) -> ndarray:
        """
        This method assumes that losses are uniform and specified in the configuration of the
        experiment (in test case setup).
        :return: Input state after losses.
        """
        lossy_input = asarray(self._initial_state)
        for i in range(len(self._initial_state)):
            for _ in range(self._initial_state[i]):
                if uniform(0, 1) < self._experiment_configuration.probability_of_uniform_loss:
                    lossy_input[i] -= 1
        return lossy_input
