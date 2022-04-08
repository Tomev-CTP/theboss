__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to test the BOBS strategy accuracy.
"""

from tests.gcc_based_strategies_tests_base import GCCBasedStrategiesTestsBase
from theboss.simulation_strategies.simulation_strategy_factory import StrategyType


# TODO TR: Add tests for lossy state approximation strategy


class TestBOBSStrategy(GCCBasedStrategiesTestsBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_exact_sampling_accuracy(self):
        self._prepare_lossless_distance_experiment_settings()

        self._sampling_tvd_experiment_config.hierarchy_level = \
            self._sampling_tvd_experiment_config.number_of_modes

        self._perform_lossless_test(StrategyType.BOBS)

    def test_exact_sampling_accuracy_with_binned_input(self):
        self._prepare_lossless_distance_experiments_settings_with_binned_inputs()

        self._sampling_tvd_experiment_config.hierarchy_level = \
            self._sampling_tvd_experiment_config.number_of_modes

        self._perform_lossless_test(StrategyType.BOBS)

    def test_exact_lossy_sampling_accuracy(self):
        self._prepare_lossy_distance_experiment_settings()

        self._sampling_tvd_experiment_config.hierarchy_level = \
            self._sampling_tvd_experiment_config.number_of_modes

        self._perform_test_for_uniform_losses(StrategyType.BOBS)

    def test_exact_lossy_sampling_accuracy_with_binned_input(self):
        self._prepare_lossy_distance_experiment_settings_with_binned_input()

        self._sampling_tvd_experiment_config.hierarchy_level = \
            self._sampling_tvd_experiment_config.number_of_modes

        self._perform_test_for_uniform_losses(StrategyType.BOBS)

    def _compute_bobs_approximation_tvd_bound(self):
        """
        Compute bounds for BOBS strategy. For details check [2], formula (22).

        :return: TVD bound for BOBS algorithm.
        """
        eta_eff = self._uniform_transmissivity
        n = self._sampling_tvd_experiment_config.initial_number_of_particles

        bound = pow(eta_eff, 2) / 2
        bound *= n - self._sampling_tvd_experiment_config.hierarchy_level
        bound += eta_eff * (1 - eta_eff) / 2

        return bound

    def test_lossy_approximate_sampling_accuracy(
            self):
        self._prepare_lossy_distance_experiment_settings()

        self._sampling_tvd_experiment_config.approximated_modes_number = \
            self._approximated_modes_number

        k = self._strategies_factory.experiment_configuration.number_of_modes
        k -= self._approximated_modes_number
        self._strategies_factory.experiment_configuration.hierarchy_level = k

        self._sampling_tvd_experiment_config.initial_state = \
            self._nonuniform_strategy_initial_state

        self._perform_test_for_uniform_losses(
            StrategyType.BOBS, self._compute_bobs_approximation_tvd_bound()
        )
