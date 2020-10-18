import unittest

from numpy import asarray

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

        distance_from_approximation_to_estimated = 2 * count_tv_distance_error_bound_of_experiment_results(
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

        self.assertLessEqual(distance_between_distributions, distance_from_approximation_to_estimated)
