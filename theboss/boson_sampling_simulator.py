__author__ = "Tomasz Rybotycki"

"""
    This file holds an implementation of generic boson sampling experiment simulator.
"""

# TODO TR: Expand the simulator to justify it's existence.
#   - Add frequencies counting
#   - ???

from typing import List, Sequence, Tuple

from .simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)


class BosonSamplingSimulator:
    def __init__(self, simulation_strategy: SimulationStrategyInterface) -> None:
        self._simulation_strategy = simulation_strategy

    def get_classical_simulation_results(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int]]:
        return self._simulation_strategy.simulate(input_state, samples_number)
