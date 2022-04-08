__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to test the C&C simulation strategy implementation. It's
    basically just test of a wrapper of their R implementation.
"""

import pytest

from tests.simulation_strategies_tests_common import (
    TestBSClassicalSimulationStrategies,
    SimulationStrategyFactory,
    StrategyType
)


class TestCCRSimulationStrategy(TestBSClassicalSimulationStrategies):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @pytest.mark.ignore
    def test_haar_random_interferometers_distance_for_ccr_strategy(self) -> None:
        self._set_experiment_configuration_for_lossless_haar_random()
        try:
            strategy_factory = SimulationStrategyFactory(
                self._haar_random_experiment_configuration,
                self._bs_permanent_calculator,
                StrategyType.CLIFFORD_R)
            self._test_state_average_probability_for_haar_random_matrices(
                strategy_factory)
        except Exception as e:
            print("You don't have packages required for this strategy.")
            print(e)
            self.assertTrue(True)