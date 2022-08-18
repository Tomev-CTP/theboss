__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to contain all the common operations done during the
    strategies tests. This way we can separate the test_simulation_strategies file
    into the distinct entities and thus reducing the complexity of analyzing them.
"""

import unittest
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple
from numpy import array, average
from numpy.random import randint
from scipy.stats import unitary_group
from theboss.permanent_calculators.bs_permanent_calculator_factory import (
    BSPermanentCalculatorFactory,
)
from theboss.simulation_strategies.simulation_strategy_factory import (
    SimulationStrategyFactory,
)
from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BosonSamplingExperimentConfiguration,
)
from theboss.distribution_calculators.bs_sample_based_distribution_calculator import (
    BSSampleBasedDistributionCalculator,
    BSDistributionCalculatorInterface,
)
from theboss.quantum_computations_utilities import (
    count_tv_distance_error_bound_of_experiment_results,
    count_total_variation_distance,
)
from theboss.simulation_strategies.simulation_strategy_factory import (
    StrategyType,
    SimulationStrategyInterface,
)
from theboss.boson_sampling_utilities import bosonic_space_dimension
from tqdm import tqdm


@dataclass
class SamplingAccuracyExperimentConfiguration:
    exact_calculator: BSDistributionCalculatorInterface
    estimation_calculator: BSSampleBasedDistributionCalculator
    approximation_tvd_bound: float
    outcomes_number: int


class TestBSClassicalSimulationStrategies(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._permutation_matrix = None

        self._number_of_samples_for_estimated_distribution_calculation = None
        self._probability_of_error_in_distribution_calculation = None

        self._distance_calculation_initial_state = None
        self._distance_calculation_binned_initial_state = None
        self._distance_calculation_number_of_particles_lost = None
        self._uniform_transmissivity = None

        self._approximated_modes_number = None

        self._sampling_tvd_experiment_config = None

        self._strategies_factory = None

        self._haar_random_matrices_number = None
        self._haar_random_experiment_input_state = None
        self._haar_random_experiment_configuration = None
        self._haar_random_binned_experiment_input_state = None

        self._strategy_initial_state = None
        self._calculator_initial_state = None

    def setUp(self) -> None:
        self._permutation_matrix = array(
            [
                [0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0],
            ],
            dtype=complex,
        )

        self._number_of_samples_for_estimated_distribution_calculation = int(1e3)
        self._probability_of_error_in_distribution_calculation = 1e-4

        self._distance_calculation_initial_state = [1, 1, 1, 1, 0]
        self._distance_calculation_binned_initial_state = [2, 1, 1, 0, 0]
        self._distance_calculation_number_of_particles_lost = 2
        self._uniform_transmissivity = 0.5

        self._approximated_modes_number = 3

        distance_calculation_initial_number_of_particles = sum(
            self._distance_calculation_initial_state
        )

        self._sampling_tvd_experiment_config = BosonSamplingExperimentConfiguration(
            interferometer_matrix=self._permutation_matrix,
            initial_state=self._distance_calculation_initial_state,
            initial_number_of_particles=distance_calculation_initial_number_of_particles,
            number_of_modes=len(self._distance_calculation_initial_state),
            number_of_particles_lost=self._distance_calculation_number_of_particles_lost,
            number_of_particles_left=distance_calculation_initial_number_of_particles
            - self._distance_calculation_number_of_particles_lost,
            uniform_transmissivity=0.8,
        )

        permanent_calculator_factory = BSPermanentCalculatorFactory(
            self._permutation_matrix, None, None
        )
        self._bs_permanent_calculator = (
            permanent_calculator_factory.generate_calculator()
        )

        self._strategies_factory = SimulationStrategyFactory(
            self._sampling_tvd_experiment_config, self._bs_permanent_calculator
        )

        self._haar_random_matrices_number = 10
        self._haar_random_experiment_input_state = [1, 1, 1, 1, 0]
        haar_random_initial_number_of_particles = sum(
            self._haar_random_experiment_input_state
        )
        haar_random_number_of_particles_lost = 2
        self._haar_random_experiment_configuration = (
            BosonSamplingExperimentConfiguration(
                interferometer_matrix=array([], dtype=complex),
                initial_state=self._haar_random_experiment_input_state,
                initial_number_of_particles=haar_random_number_of_particles_lost,
                number_of_modes=len(self._haar_random_experiment_input_state),
                number_of_particles_lost=haar_random_number_of_particles_lost,
                number_of_particles_left=haar_random_initial_number_of_particles
                - haar_random_number_of_particles_lost,
            )
        )
        self._haar_random_binned_experiment_input_state = [2, 1, 1, 1, 0]

        self._calculator_initial_state = self._distance_calculation_initial_state

    def _prepare_lossless_distance_experiments_settings_with_binned_inputs(
        self,
    ) -> None:
        self._sampling_tvd_experiment_config.initial_state = (
            self._distance_calculation_binned_initial_state
        )
        self._strategy_initial_state = self._distance_calculation_binned_initial_state
        initial_number_of_particles = sum(
            self._distance_calculation_binned_initial_state
        )
        self._sampling_tvd_experiment_config.initial_number_of_particles = (
            initial_number_of_particles
        )
        self._sampling_tvd_experiment_config.number_of_particles_lost = 0
        self._sampling_tvd_experiment_config.number_of_particles_left = (
            initial_number_of_particles
        )
        self._sampling_tvd_experiment_config.uniform_transmissivity = 1

    def _prepare_lossy_distance_experiment_settings_with_binned_input(self):
        self._sampling_tvd_experiment_config.initial_state = (
            self._distance_calculation_binned_initial_state
        )
        self._strategy_initial_state = self._distance_calculation_binned_initial_state
        initial_number_of_particles = sum(self._distance_calculation_initial_state)
        self._sampling_tvd_experiment_config.initial_number_of_particles = (
            initial_number_of_particles
        )
        self._sampling_tvd_experiment_config.number_of_particles_lost = (
            self._distance_calculation_number_of_particles_lost
        )
        self._sampling_tvd_experiment_config.number_of_particles_left = (
            initial_number_of_particles
            - self._distance_calculation_number_of_particles_lost
        )
        self._sampling_tvd_experiment_config.uniform_transmissivity = (
            self._uniform_transmissivity
        )
        self._calculator_initial_state = self._distance_calculation_binned_initial_state

    def _prepare_lossy_distance_experiment_settings(self) -> None:
        # At least for now lossy experiments are only for standard input states.
        self._sampling_tvd_experiment_config.initial_state = (
            self._distance_calculation_initial_state
        )
        self._strategy_initial_state = self._distance_calculation_initial_state
        initial_number_of_particles = sum(self._distance_calculation_initial_state)
        self._sampling_tvd_experiment_config.initial_number_of_particles = (
            initial_number_of_particles
        )
        self._sampling_tvd_experiment_config.number_of_particles_lost = (
            self._distance_calculation_number_of_particles_lost
        )
        self._sampling_tvd_experiment_config.number_of_particles_left = (
            initial_number_of_particles
            - self._distance_calculation_number_of_particles_lost
        )
        self._sampling_tvd_experiment_config.uniform_transmissivity = (
            self._uniform_transmissivity
        )

    def _check_if_approximation_is_within_bounds(
        self,
        sampling_accuracy_experiment_config: SamplingAccuracyExperimentConfiguration,
    ) -> None:

        distance_from_exact_to_estimated = self._compute_tvd_between_distribution_and_frequencies(
            exact_distribution_calculator=sampling_accuracy_experiment_config.exact_calculator,
            estimated_distribution_calculator=sampling_accuracy_experiment_config.estimation_calculator,
        )

        distance_from_approximation_to_estimated = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=sampling_accuracy_experiment_config.outcomes_number,
            samples_number=self._number_of_samples_for_estimated_distribution_calculation,
            error_probability=self._probability_of_error_in_distribution_calculation,
        )

        # Using triangle inequality of (TV) distance.
        max_allowed_distance = (
            sampling_accuracy_experiment_config.approximation_tvd_bound
            + distance_from_approximation_to_estimated
        )

        self.assertLessEqual(
            distance_from_exact_to_estimated,
            max_allowed_distance,
            f"Distance from exact distribution ({distance_from_exact_to_estimated}) is "
            f"greater than maximum distance allowed ({max_allowed_distance}).",
        )

    def _generate_frequencies_calculator(
        self,
        strategy: SimulationStrategyInterface,
        outcomes: List[Tuple[int, ...]] = None,
    ) -> BSSampleBasedDistributionCalculator:
        estimated_distribution_calculator = BSSampleBasedDistributionCalculator(
            experiment_configuration=self._sampling_tvd_experiment_config,
            strategy=strategy,
            outcomes=outcomes,
        )
        return estimated_distribution_calculator

    def _compute_tvd_between_distribution_and_frequencies(
        self,
        exact_distribution_calculator: BSDistributionCalculatorInterface,
        estimated_distribution_calculator: BSSampleBasedDistributionCalculator,
    ) -> float:
        """
        Using specified calculators, it counts exact and estimated probability
        distributions and computes the tv distance between them.

        :param exact_distribution_calculator:       Calculator of the exact (ideal,
                                                    bruteforce permanents) distribution.
        :param estimated_distribution_calculator:   Calculator of the approximate
                                                    (sampling according to strategy)
                                                    distribution.

        :return: TV distance between exact and approximated distribution.
        """

        exact_distribution = exact_distribution_calculator.calculate_distribution()

        estimated_distribution_calculator.outcomes = (
            exact_distribution_calculator.get_outcomes_in_proper_order()
        )

        if (
            self._strategies_factory.strategy_type == StrategyType.GCC_GENERAL_LOSSES
            or self._strategies_factory.strategy_type == StrategyType.GMF_GENERAL_LOSSES
        ):
            self._strategies_factory.bs_permanent_calculator.matrix *= pow(
                self._uniform_transmissivity, 0.5
            )

        approximated_distribution = estimated_distribution_calculator.calculate_approximate_distribution(
            samples_number=self._number_of_samples_for_estimated_distribution_calculation
        )

        return count_total_variation_distance(
            exact_distribution, approximated_distribution
        )

    def _set_experiment_configuration_for_lossless_haar_random(self) -> None:
        self._haar_random_experiment_configuration.initial_state = (
            self._haar_random_experiment_input_state
        )
        number_of_particles_in_the_experiment = sum(
            self._haar_random_experiment_input_state
        )
        self._haar_random_experiment_configuration.initial_number_of_particles = (
            number_of_particles_in_the_experiment
        )
        self._haar_random_experiment_configuration.number_of_particles_left = (
            number_of_particles_in_the_experiment
        )

    def _prepare_lossless_distance_experiment_settings(self) -> None:
        self._sampling_tvd_experiment_config.initial_state = (
            self._distance_calculation_initial_state
        )
        initial_number_of_particles = sum(self._distance_calculation_initial_state)
        self._sampling_tvd_experiment_config.initial_number_of_particles = (
            initial_number_of_particles
        )
        self._sampling_tvd_experiment_config.number_of_particles_lost = 0
        self._sampling_tvd_experiment_config.number_of_particles_left = (
            initial_number_of_particles
        )
        self._sampling_tvd_experiment_config.uniform_transmissivity = 1

    def _test_state_average_probability_for_haar_random_matrices(
        self, strategy_factory: SimulationStrategyFactory
    ) -> None:
        number_of_outcomes = bosonic_space_dimension(
            particles_number=self._haar_random_experiment_configuration.number_of_particles_left,
            modes_number=self._haar_random_experiment_configuration.number_of_modes,
        )

        error_bound = count_tv_distance_error_bound_of_experiment_results(
            outcomes_number=number_of_outcomes,
            samples_number=self._haar_random_matrices_number,
            error_probability=self._probability_of_error_in_distribution_calculation,
        )

        probabilities_list = []
        current_probabilities = []

        for _ in tqdm(range(self._haar_random_matrices_number)):

            experiment_configuration = deepcopy(
                self._haar_random_experiment_configuration
            )
            experiment_configuration.interferometer_matrix = unitary_group.rvs(
                self._haar_random_experiment_configuration.number_of_modes
            )
            strategy_factory.experiment_configuration = experiment_configuration
            self._bs_permanent_calculator.matrix = (
                experiment_configuration.interferometer_matrix
            )
            strategy_factory.bs_permanent_calculator = self._bs_permanent_calculator
            distribution_calculator = BSSampleBasedDistributionCalculator(
                experiment_configuration, strategy_factory.generate_strategy()
            )

            current_probabilities = distribution_calculator.calculate_approximate_distribution(
                samples_number=self._number_of_samples_for_estimated_distribution_calculation
            )

            if len(probabilities_list) == 0:
                probabilities_list = [[] for _ in range(len(current_probabilities))]

            for j in range(len(current_probabilities)):
                probabilities_list[j].append(current_probabilities[j])

        random_outcome_index = randint(0, len(current_probabilities))
        self.assertAlmostEqual(
            number_of_outcomes ** (-1),
            average(probabilities_list[random_outcome_index]),
            delta=error_bound,
        )
