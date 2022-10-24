__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to test the accuracy of the uniformly lossy generalized
    mean field strategy. This script tests only simulations with uniform or no losses.
    It also tests several different hierarchy levels.
"""

from tests.gcc_based_strategies_tests_base import GCCBasedStrategiesTestsBase
from theboss.simulation_strategies.simulation_strategy_factory import StrategyType


class TestUUniformlyLossyGMFStrategy(GCCBasedStrategiesTestsBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_exact_sampling_accuracy(self):
        self._prepare_lossless_distance_experiment_settings()

        self._sampling_tvd_experiment_config.hierarchy_level = (
            self._sampling_tvd_experiment_config.number_of_modes
        )

        self._perform_lossless_test(StrategyType.GMF_UNIFORM_LOSSES)

    def test_exact_sampling_accuracy_with_binned_input(self):
        self._prepare_lossless_distance_experiments_settings_with_binned_inputs()

        self._sampling_tvd_experiment_config.hierarchy_level = (
            self._sampling_tvd_experiment_config.number_of_modes
        )

        self._perform_lossless_test(StrategyType.GMF_UNIFORM_LOSSES)

    def test_exact_lossy_sampling_accuracy(self):
        self._prepare_lossy_distance_experiment_settings()

        self._sampling_tvd_experiment_config.hierarchy_level = (
            self._sampling_tvd_experiment_config.number_of_modes
        )

        self._perform_test_for_uniform_losses(StrategyType.GMF_UNIFORM_LOSSES)

    def test_exact_lossy_sampling_accuracy_with_binned_input(self):
        self._prepare_lossy_distance_experiment_settings_with_binned_input()

        self._sampling_tvd_experiment_config.hierarchy_level = (
            self._sampling_tvd_experiment_config.number_of_modes
        )

        self._perform_test_for_uniform_losses(StrategyType.GMF_UNIFORM_LOSSES)

    def _compute_gmf_approximation_tvd_bound(self):
        """
        Compute bounds for GMF strategy. For details check [2], formula (22).

        :return:
            TVD upper bound for GMF algorithm.
        """
        eta_eff = self._uniform_transmission_probability
        n = self._sampling_tvd_experiment_config.initial_number_of_particles

        bound = pow(eta_eff, 2) / 2
        bound *= n - self._sampling_tvd_experiment_config.hierarchy_level
        bound += eta_eff * (1 - eta_eff) / 2

        return bound

    def test_lossy_approximate_sampling_accuracy(self):
        self._prepare_lossy_distance_experiment_settings()

        self._sampling_tvd_experiment_config.approximated_modes_number = (
            self._approximated_modes_number
        )

        k = (
            self._strategies_factory.experiment_configuration.number_of_modes
            - self._approximated_modes_number
        )
        self._strategies_factory.experiment_configuration.hierarchy_level = k

        self._perform_test_for_uniform_losses(
            StrategyType.GMF_UNIFORM_LOSSES,
            self._compute_gmf_approximation_tvd_bound(),
        )

    def test_approximate_sampling_accuracy_with_low_hierarchy_level(self) -> None:
        self._prepare_lossy_distance_experiment_settings()

        self._strategies_factory.experiment_configuration.hierarchy_level = 1

        self._sampling_tvd_experiment_config.approximated_modes_number = (
            self._sampling_tvd_experiment_config.number_of_modes
            - self._sampling_tvd_experiment_config.hierarchy_level
        )

        self._perform_test_for_uniform_losses(
            StrategyType.GMF_UNIFORM_LOSSES,
            self._compute_gmf_approximation_tvd_bound(),
        )

    def test_approximate_sampling_accuracy_with_medium_hierarchy_level(self) -> None:
        self._prepare_lossy_distance_experiment_settings()

        self._strategies_factory.experiment_configuration.hierarchy_level = (
            self._sampling_tvd_experiment_config.number_of_modes // 2
        )

        self._sampling_tvd_experiment_config.approximated_modes_number = (
            self._sampling_tvd_experiment_config.number_of_modes
            - self._sampling_tvd_experiment_config.hierarchy_level
        )

        self._perform_test_for_uniform_losses(
            StrategyType.GMF_UNIFORM_LOSSES,
            self._compute_gmf_approximation_tvd_bound(),
        )

    def test_approximate_sampling_accuracy_with_all_approximate_modes(self) -> None:
        self._prepare_lossy_distance_experiment_settings()

        self._strategies_factory.experiment_configuration.hierarchy_level = 0

        self._sampling_tvd_experiment_config.approximated_modes_number = (
            self._sampling_tvd_experiment_config.number_of_modes
        )

        self._perform_test_for_uniform_losses(
            StrategyType.GMF_UNIFORM_LOSSES,
            self._compute_gmf_approximation_tvd_bound(),
        )