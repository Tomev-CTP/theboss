__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to provide a base class for GCC related strategies
    tests. It has to be made separate, because subclassing a class with tests methods
    will make pytest run all of the tests, from both sub and superclass.
"""

from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    bosonic_space_dimension,
    lossy_bosonic_space_dimension,
)
from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithFixedLosses,
    BSDistributionCalculatorWithUniformLosses,
)
from tests.simulation_strategies_tests_common import (
    TestBSClassicalSimulationStrategies,
    SamplingAccuracyExperimentConfiguration,
)
from theboss.simulation_strategies.simulation_strategy_factory import StrategyType


class GCCBasedStrategiesTestsBase(TestBSClassicalSimulationStrategies):
    def _perform_lossless_test(self, strategy: StrategyType = StrategyType.GCC) -> None:
        self._strategies_factory.experiment_configuration = (
            self._sampling_tvd_experiment_config
        )
        self._strategies_factory.strategy_type = strategy
        distance_experiment_configuration = SamplingAccuracyExperimentConfiguration(
            # This exact calculator, when there are no losses, will do the work just fine.
            exact_calculator=BSDistributionCalculatorWithFixedLosses(
                self._sampling_tvd_experiment_config, self._bs_permanent_calculator
            ),
            estimation_calculator=self._generate_frequencies_calculator(
                self._strategies_factory.generate_strategy()
            ),
            outcomes_number=bosonic_space_dimension(
                particles_number=self._sampling_tvd_experiment_config.number_of_particles_left,
                modes_number=self._sampling_tvd_experiment_config.number_of_modes,
            ),
            approximation_tvd_bound=0,  # This strategy returns exact solution.
        )
        self._check_if_approximation_is_within_bounds(distance_experiment_configuration)

    def _perform_test_for_uniform_losses(
        self,
        strategy: StrategyType = StrategyType.UNIFORM_LOSSES_GCC,
        approximation_bound: int = 0,
    ) -> None:

        self._strategies_factory.experiment_configuration = (
            self._sampling_tvd_experiment_config
        )

        self._strategies_factory.strategy_type = strategy

        self._sampling_tvd_experiment_config.initial_state = (
            self._calculator_initial_state
        )

        exact_calculator = BSDistributionCalculatorWithUniformLosses(
            self._sampling_tvd_experiment_config, self._bs_permanent_calculator
        )

        self._sampling_tvd_experiment_config.initial_state = (
            self._strategy_initial_state
        )

        if strategy == StrategyType.LOSSY_NET_GCC or strategy == StrategyType.BOBS:
            self._strategies_factory.bs_permanent_calculator.matrix *= pow(
                self._uniform_transmissivity, 0.5
            )

        distance_experiment_configuration = SamplingAccuracyExperimentConfiguration(
            # This exact calculator, when there are no losses, will do the work just fine.
            exact_calculator=exact_calculator,
            estimation_calculator=self._generate_frequencies_calculator(
                self._strategies_factory.generate_strategy(),
                outcomes=exact_calculator.get_outcomes_in_proper_order(),
            ),
            outcomes_number=lossy_bosonic_space_dimension(
                maximal_particles_number=self._sampling_tvd_experiment_config.initial_number_of_particles,
                modes_number=self._sampling_tvd_experiment_config.number_of_modes,
            ),
            approximation_tvd_bound=approximation_bound,
        )

        self._check_if_approximation_is_within_bounds(distance_experiment_configuration)
