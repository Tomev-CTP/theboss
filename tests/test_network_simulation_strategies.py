__author__ = "Tomasz Rybotycki"

import unittest
from copy import deepcopy
from random import uniform
from typing import List, Union, Tuple

from numpy import eye, ndarray
from scipy.stats import unitary_group

from theboss.boson_sampling_simulator import BosonSamplingSimulator
from theboss.permanent_calculators.bs_permanent_calculator_factory import (
    BSPermanentCalculatorFactory,
)
from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithUniformLosses,
    BosonSamplingExperimentConfiguration,
)
from theboss.distribution_calculators.bs_sample_based_distribution_calculator import (
    BSSampleBasedDistributionCalculator,
)
from theboss.network_simulation_strategy.lossy_network_simulation_strategy import (
    LossyNetworkSimulationStrategy,
)
from theboss.quantum_computations_utilities import (
    count_total_variation_distance,
    count_tv_distance_error_bound_of_experiment_results,
)
from theboss.simulation_strategies.simulation_strategy_factory import (
    StrategyType,
    SimulationStrategyFactory,
)


class TestBosonSamplingClassicalSimulationStrategies(unittest.TestCase):
    def setUp(self) -> None:
        uniform_transmissivity: float = 0.5

        self._initial_state: List[int] = [1, 1, 1, 1, 0, 0]
        self._number_of_samples_for_experiments: int = 1000
        self._probability_of_error_in_distribution_calculation: float = 0.001

        random_unitary = unitary_group.rvs(len(self._initial_state))

        self._interferometer_matrix = random_unitary
        self._lossy_interferometer_matrix = uniform_transmissivity * random_unitary

        self._experiment_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._lossy_interferometer_matrix,
            initial_state=self._initial_state,
            initial_number_of_particles=sum(self._initial_state),
            number_of_modes=len(self._initial_state),
            number_of_particles_lost=0,  # Losses should only come from network.
            number_of_particles_left=sum(self._initial_state),
            uniform_transmissivity=uniform_transmissivity,
            network_simulation_strategy=LossyNetworkSimulationStrategy(
                self._lossy_interferometer_matrix
            ),
        )

        permanent_calculator_factory = BSPermanentCalculatorFactory(
            self._lossy_interferometer_matrix, None, None
        )

        self._permanent_calculator = permanent_calculator_factory.generate_calculator()
        self._strategy_factory = SimulationStrategyFactory(
            self._experiment_configuration, self._permanent_calculator
        )

        calculator = BSDistributionCalculatorWithUniformLosses(
            self._experiment_configuration, self._permanent_calculator
        )
        self._possible_outcomes: List[
            Tuple[int, ...]
        ] = calculator.get_outcomes_in_proper_order()
        self._possible_outcomes_number: int = len(self._possible_outcomes)

        self._tvd_bound_between_estimated_distributions = self._compute_statistical_bound_on_two_approximate_distributions_tvd(
            outcomes_number=self._possible_outcomes_number
        )

    def test_lossy_network_simulation_number_of_particles(self) -> None:
        """
        This test checks if number of particles for lossy network simulator is lower
        than number of particles without the losses. Note that I can be pretty sure it 
        will be lower in large losses' regime, but in case of lower losses this test may
        not hold.
        """
        self._strategy_factory.strategy_type = StrategyType.FIXED_LOSS

        strategy = self._strategy_factory.generate_strategy()
        simulator = BosonSamplingSimulator(strategy)
        lossy_average_number_of_particles = 0
        samples = simulator.get_classical_simulation_results(
            self._initial_state, self._number_of_samples_for_experiments,
        )
        for sample in samples:
            lossy_average_number_of_particles += sum(sample)

        lossy_average_number_of_particles /= self._number_of_samples_for_experiments
        lossless_number_of_particles = sum(self._initial_state)
        self.assertLess(lossy_average_number_of_particles, lossless_number_of_particles)

    def test_lossy_network_with_uniform_losses_distribution_accuracy(self) -> None:
        """
        This test is for uniform losses. It checks if uniform losses in network
        simulations are close to the simulation with uniform losses at the input.
        """
        self._strategy_factory.strategy_type = StrategyType.UNIFORM_LOSS

        estimated_distribution_calculator = BSSampleBasedDistributionCalculator(
            experiment_configuration=self._experiment_configuration,
            strategy=self._strategy_factory.generate_strategy(),
            outcomes=self._possible_outcomes,
        )

        uniform_losses_distribution = (
            estimated_distribution_calculator.calculate_approximate_distribution()
        )

        self._check_if_given_distribution_is_close_to_lossy_network_distribution(
            uniform_losses_distribution
        )

    def _check_if_given_distribution_is_close_to_lossy_network_distribution(
        self, distribution: Union[ndarray, List[float]]
    ) -> None:
        """
        Common part of distribution comparison tests.

        :param distribution: Given distribution to compare with lossy distribution.
        """

        self._experiment_configuration.interferometer_matrix = (
            self._lossy_interferometer_matrix
        )

        estimated_distribution_calculator = BSSampleBasedDistributionCalculator(
            experiment_configuration=self._experiment_configuration,
            strategy=self._strategy_factory.generate_strategy(),
            outcomes=self._possible_outcomes,
        )

        lossy_network_distribution = (
            estimated_distribution_calculator.calculate_approximate_distribution()
        )

        distance_between_distributions = count_total_variation_distance(
            lossy_network_distribution, distribution
        )

        self.assertLessEqual(
            distance_between_distributions,
            self._tvd_bound_between_estimated_distributions,
        )

    def _compute_statistical_bound_on_two_approximate_distributions_tvd(
        self, outcomes_number: int
    ) -> float:
        """
        Compute the statistical bound on TVD between two frequencies sets. The "2"
        coefficient comes from the fact that two distributions are compared and (at
        least up to my understanding) in the worst case scenario they can be away
        from the theoretical distribution by a given bound, but in a "different
        direction".

        :param outcomes_number:


        :return:
        """
        return 2 * count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=outcomes_number,
            samples_number=self._number_of_samples_for_experiments,
            error_probability=self._probability_of_error_in_distribution_calculation,
        )

    def test_lossy_network_with_uniform_losses_distribution_accuracy_against_gcc(
        self,
    ) -> None:
        """
        This test checks if lossy networks distribution provides same approximate (up
        to the statistical bounds) distribution as generalized Clifford & Clifford
        distribution with lossy inputs.
        """
        generalized_cliffords_distribution = (
            self._calculate_gcc_distribution_with_lossy_inputs()
        )

        self._strategy_factory.strategy_type = StrategyType.LOSSY_NET_GCC

        self._check_if_given_distribution_is_close_to_lossy_network_distribution(
            generalized_cliffords_distribution
        )

    def _calculate_gcc_distribution_with_lossy_inputs(self) -> List[float]:
        """
        This method calculates approximate distribution for lossy states using
        GC&C method.

        :return:
            Approximate distribution.
        """
        self._strategy_factory.strategy_type = StrategyType.GCC
        self._permanent_calculator.matrix = self._interferometer_matrix
        strategy = self._strategy_factory.generate_strategy()
        simulator = BosonSamplingSimulator(strategy)

        samples = []

        for _ in range(self._number_of_samples_for_experiments):
            samples.extend(
                simulator.get_classical_simulation_results(
                    self._get_uniformly_lossy_input_state(), 1
                )
            )

        return self._compute_distribution(samples, self._possible_outcomes)

    @staticmethod
    def _compute_distribution(
        samples: List[Tuple[int, ...]], possible_outcomes: List[Tuple[int, ...]]
    ) -> List[float]:
        probabilities = [0] * len(possible_outcomes)

        for sample in samples:
            probabilities[possible_outcomes.index(sample)] += 1

        for i in range(len(probabilities)):
            probabilities[i] /= len(samples)

        return probabilities

    def _get_uniformly_lossy_input_state(self) -> List[int]:
        """
        This method assumes that losses are uniform and specified in the configuration
        of the experiment (in test case setup).

        :return: Input state after losses.
        """
        lossy_input = deepcopy(self._initial_state)
        for i in range(len(self._initial_state)):
            for _ in range(self._initial_state[i]):
                if (
                    uniform(0, 1)
                    > self._experiment_configuration.uniform_transmissivity
                ):
                    lossy_input[i] -= 1
        return lossy_input

    def test_uniformly_lossy_network_on_occupied_modes_and_higher_losses_on_empty_modes(
        self,
    ) -> None:
        """
        This test checks the situation, for which losses are uniform only for
        these modes that have any bosons in it, empty modes' transmissivity is set to 0.
        The results should be identical (up to statistical error) for both cases.
        """
        experiment_configuration = deepcopy(self._experiment_configuration)
        updated_interferometer_matrix = experiment_configuration.interferometer_matrix
        update_matrix = eye(len(self._initial_state), dtype=complex)

        for i in range(len(self._initial_state)):
            if self._initial_state == 0:
                update_matrix[i][i] = 0

        updated_interferometer_matrix = update_matrix @ updated_interferometer_matrix

        experiment_configuration.interferometer_matrix = updated_interferometer_matrix

        self._strategy_factory.strategy_type = StrategyType.FIXED_LOSS
        self._strategy_factory.experiment_configuration = experiment_configuration

        estimated_distribution_calculator = BSSampleBasedDistributionCalculator(
            experiment_configuration=self._experiment_configuration,
            strategy=self._strategy_factory.generate_strategy(),
            outcomes=self._possible_outcomes,
        )

        distribution_with_huge_losses_on_empty_modes = (
            estimated_distribution_calculator.calculate_approximate_distribution()
        )

        self._strategy_factory.experiment_configuration = self._experiment_configuration
        self._check_if_given_distribution_is_close_to_lossy_network_distribution(
            distribution_with_huge_losses_on_empty_modes
        )

    def test_distance_of_gcc_with_lossy_network_and_lossy_input(self) -> None:
        """
        Test if the simulation using lossy net and the simulation using lossy input
        yields the same results.
        """
        distribution_with_lossy_net = (
            self._calculate_gcc_distribution_with_lossy_network()
        )

        distribution_with_lossy_input = (
            self._calculate_gcc_distribution_with_lossy_inputs()
        )

        distance_between_distributions = count_total_variation_distance(
            distribution_with_lossy_input, distribution_with_lossy_net
        )

        self.assertLessEqual(
            distance_between_distributions,
            self._tvd_bound_between_estimated_distributions,
        )

    def _calculate_gcc_distribution_with_lossy_network(self) -> List[float]:
        """
        This method calculates approximate distribution for lossy states using
        generalized Clifford & Clifford method.

        :return: Approximate distribution.
        """
        self._strategy_factory.strategy_type = StrategyType.LOSSY_NET_GCC
        self._strategy_factory.experiment_configuration.interferometer_matrix = (
            self._lossy_interferometer_matrix
        )
        strategy = self._strategy_factory.generate_strategy()
        simulator = BosonSamplingSimulator(strategy)
        samples = simulator.get_classical_simulation_results(
            self._initial_state, self._number_of_samples_for_experiments,
        )

        return self._compute_distribution(samples, self._possible_outcomes)
