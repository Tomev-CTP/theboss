__author__ = "Tomasz Rybotycki"

"""
    The intention behind this code is to have tests for different simulation strategies
    that for various reasons aren't important for the ongoing research anymore.
"""

from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    lossy_bosonic_space_dimension,
    bosonic_space_dimension,
)
from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithUniformLosses,
    BSDistributionCalculatorWithFixedLosses,
)
from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BSDistributionCalculatorInterface,
)
from theboss.simulation_strategies.simulation_strategy_factory import (
    StrategyType,
    SimulationStrategyFactory,
)
from tests.simulation_strategies_tests_common import (
    TestBSClassicalSimulationStrategies,
    SamplingAccuracyExperimentConfiguration,
)
from scipy.special import binom
from math import factorial


class TestFullApproximationBSSimulationStrategies(TestBSClassicalSimulationStrategies):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()

    def _set_experiment_configuration_for_standard_haar_random(self) -> None:
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
            - self._haar_random_experiment_configuration.number_of_particles_lost
        )

    @staticmethod
    def _compute_fixed_losses_approximation_tvd_bound(
        initial_number_of_particles: int, number_of_particles_left: int
    ) -> float:
        """
            This is the distance bound between experimental and ideal results for
            BosonSampling with fixed losses. Basically this is capital Delta from [1].
            Using n and l notation from the paper [1] for readability purposes.
            This will be used for uniform losses and fixed losses bound calculation.

            :return:    Distance bound between experimental and ideal results for
                        fixed losses.
        """
        n = initial_number_of_particles
        l = number_of_particles_left
        error_bound = 1.0 - (factorial(n) / (pow(n, l) * factorial(n - l)))
        return error_bound

    def _compute_uniform_loss_approximation_tvd_bound(self) -> float:
        """
        This is the distance bound between approximated and ideal results for the
        BosonSampling with uniform losses. Basically this is capital Delta from [1].
        Using eta, n and l notation from the paper [1] for readability purposes.

        :return:    Distance bound between the experimental and the ideal distribution
                    for uniformly lossy BS.
        """
        #
        error_bound = 0
        n = self._sampling_tvd_experiment_config.initial_number_of_particles
        eta = self._sampling_tvd_experiment_config.uniform_transmissivity
        for number_of_particles_left in range(n + 1):
            l = number_of_particles_left
            subdistribution_weight = pow(eta, l) * binom(n, l) * pow(1.0 - eta, n - l)
            error_bound += (
                subdistribution_weight
                * self._compute_fixed_losses_approximation_tvd_bound(n, l)
            )
        return error_bound

    def _perform_full_approximation_strategies_test(
        self,
        exact_calculator: BSDistributionCalculatorInterface,
        tvd_bound: float,
        outcomes_number: int,
    ) -> None:

        self._strategies_factory.experiment_configuration = (
            self._sampling_tvd_experiment_config
        )

        distance_experiment_configuration = SamplingAccuracyExperimentConfiguration(
            exact_calculator=exact_calculator,
            estimation_calculator=self._generate_frequencies_calculator(
                self._strategies_factory.generate_strategy(),
                outcomes=exact_calculator.get_outcomes_in_proper_order(),
            ),
            outcomes_number=outcomes_number,
            approximation_tvd_bound=tvd_bound,
        )
        self._check_if_approximation_is_within_bounds(distance_experiment_configuration)

    def test_sampling_accuracy_for_fixed_losses_strategy(self) -> None:
        """
        Checks if the tvd between ideal distribution and the experimental
        frequencies is within the assumed bounds for fixed losses' strategy.

        :return None:
        """
        self._prepare_lossy_distance_experiment_settings()

        self._strategies_factory.strategy_type = StrategyType.FIXED_LOSS

        exact_calculator = BSDistributionCalculatorWithFixedLosses(
            self._sampling_tvd_experiment_config, self._bs_permanent_calculator
        )

        tvd_bound = self._compute_fixed_losses_approximation_tvd_bound(
            self._sampling_tvd_experiment_config.initial_number_of_particles,
            self._sampling_tvd_experiment_config.number_of_particles_left,
        )

        outcomes_number = bosonic_space_dimension(
            particles_number=self._sampling_tvd_experiment_config.number_of_particles_left,
            modes_number=self._sampling_tvd_experiment_config.number_of_modes,
        )

        self._perform_full_approximation_strategies_test(
            exact_calculator, tvd_bound, outcomes_number
        )

    def test_distribution_accuracy_for_uniform_losses_strategy(self) -> None:
        """
        Checks if the tvd between ideal distribution and the experimental
        frequencies is within the assumed bounds for uniform losses' strategy.

        :return None:
        """
        self._prepare_lossy_distance_experiment_settings()

        self._strategies_factory.strategy_type = StrategyType.UNIFORM_LOSS

        exact_calculator = BSDistributionCalculatorWithUniformLosses(
            self._sampling_tvd_experiment_config, self._bs_permanent_calculator
        )

        tvd_bound = self._compute_uniform_loss_approximation_tvd_bound()

        outcomes_number = lossy_bosonic_space_dimension(
            maximal_particles_number=self._sampling_tvd_experiment_config.number_of_particles_left,
            modes_number=self._sampling_tvd_experiment_config.number_of_modes,
        )

        self._perform_full_approximation_strategies_test(
            exact_calculator, tvd_bound, outcomes_number
        )

    # Haar Random tests
    def test_fixed_losses_state_average_probability_for_haar_random_matrices(
        self,
    ) -> None:
        self._set_experiment_configuration_for_standard_haar_random()

        strategy_factory = SimulationStrategyFactory(
            self._haar_random_experiment_configuration,
            self._bs_permanent_calculator,
            StrategyType.FIXED_LOSS,
        )

        self._test_state_average_probability_for_haar_random_matrices(strategy_factory)

    def test_uniform_losses_state_average_probability_for_haar_random_matrices(
        self,
    ) -> None:
        self._set_experiment_configuration_for_standard_haar_random()

        strategy_factory = SimulationStrategyFactory(
            self._haar_random_experiment_configuration,
            self._bs_permanent_calculator,
            StrategyType.UNIFORM_LOSS,
        )

        self._test_state_average_probability_for_haar_random_matrices(strategy_factory)
