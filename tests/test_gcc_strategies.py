__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to test the Generalized Clifford & Clifford strategies
    implementations.
"""


from tests.gcc_based_strategies_tests_base import GCCBasedStrategiesTestsBase
from theboss.simulation_strategies.simulation_strategy_factory import (
    StrategyType, SimulationStrategyFactory
)


class TestGCCStrategies(GCCBasedStrategiesTestsBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # GCC
    def test_exact_sampling_accuracy(self) -> None:
        """
        Checks if the distance between ideal distribution and the experimental
        frequencies is within the assumed bounds for generalized Clifford & Clifford
        strategy.

        :return None:
        """
        self._prepare_lossless_distance_experiment_settings()
        self._perform_lossless_test()

    def test_exact_sampling_accuracy_with_binned_input(
            self) -> None:
        self._prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self._perform_lossless_test()

    # GCC lossy network
    def test_exact_sampling_accuracy_for_lossy_network_strategy(self) -> None:
        self._prepare_lossless_distance_experiment_settings()
        self._perform_lossless_test(StrategyType.LOSSY_NET_GCC)

    def test_exact_sampling_accuracy_for_lossy_network_with_binned_input(self) -> None:
        self._prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self._perform_lossless_test(StrategyType.LOSSY_NET_GCC)

    def test_lossy_sampling_accuracy_for_lossy_network(self) -> None:
        self._prepare_lossy_distance_experiment_settings()
        self._perform_test_for_uniform_losses(StrategyType.LOSSY_NET_GCC)

    def test_lossy_sampling_accuracy_for_lossy_network_with_binned_input(self) -> None:
        self._prepare_lossy_distance_experiment_settings_with_binned_input()
        self._perform_test_for_uniform_losses(StrategyType.LOSSY_NET_GCC)

    # Uniform lossy GCC
    def test_exact_sampling_accuracy_for_u_losses_strategy(self) -> None:
        self._prepare_lossless_distance_experiment_settings()
        self._perform_lossless_test(StrategyType.UNIFORM_LOSSES_GCC)

    def test_exact_sampling_accuracy_for_u_losses_strategy_with_binned_input(self)\
            -> None:
        self._prepare_lossless_distance_experiments_settings_with_binned_inputs()
        self._perform_lossless_test(StrategyType.UNIFORM_LOSSES_GCC)

    def test_lossy_sampling_accuracy_for_u_losses_strategy(self) -> None:
        self._prepare_lossy_distance_experiment_settings()
        self._perform_test_for_uniform_losses(StrategyType.UNIFORM_LOSSES_GCC)

    def test_lossy_sampling_accuracy_for_u_losses_with_binned_input(self) -> None:
        self._prepare_lossy_distance_experiment_settings_with_binned_input()
        self._perform_test_for_uniform_losses(StrategyType.UNIFORM_LOSSES_GCC)

    # Haar random matrices tests
    def _set_experiment_configuration_for_binned_haar_random(self) -> None:
        # Note that binned configuration is also for lossless case (for now)
        self._haar_random_experiment_configuration.initial_state = \
            self._haar_random_binned_experiment_input_state
        number_of_particles_in_the_experiment = len(
            self._haar_random_binned_experiment_input_state)
        self._haar_random_experiment_configuration.initial_number_of_particles = \
            number_of_particles_in_the_experiment
        self._haar_random_experiment_configuration.number_of_particles_left = \
            number_of_particles_in_the_experiment

    def test_gcc_state_average_probability_for_haar_random_matrices(self) -> None:
        self._set_experiment_configuration_for_lossless_haar_random()
        strategy_factory = SimulationStrategyFactory(
            self._haar_random_experiment_configuration,
            self._bs_permanent_calculator,
            StrategyType.GCC)
        self._test_state_average_probability_for_haar_random_matrices(
            strategy_factory)

    def test_binned_input_gcc_state_average_probability_for_haar_random_matrices(self)\
            -> None:
        self._set_experiment_configuration_for_binned_haar_random()
        strategy_factory = SimulationStrategyFactory(
            self._haar_random_experiment_configuration,
            self._bs_permanent_calculator,
            StrategyType.GCC)
        self._test_state_average_probability_for_haar_random_matrices(
            strategy_factory)
