__author__ = "Tomasz Rybotycki"

"""
        
"""

from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)

from typing import List, Tuple, Sequence
from theboss.boson_sampling_utilities import generate_possible_states
from numpy.random import randint


class UniformSamplingStrategy(SimulationStrategyInterface):
    """
        A class for uniform sampling from the proper BS output states.
    """

    def __init__(self) -> None:
        pass

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:

        possible_output_states: List[Tuple[int, ...]] = generate_possible_states(
            sum(input_state), len(input_state)
        )

        return [
            possible_output_states[randint(0, len(possible_output_states))]
            for _ in range(samples_number)
        ]
