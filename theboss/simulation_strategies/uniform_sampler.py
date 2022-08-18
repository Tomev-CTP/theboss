__author__ = "Tomasz Rybotycki"

"""
This script contains an implementation of a class for uniform sampling of the BS output
states basing on the input states. It has been implemented for the tests and validators.
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
        """
        Draws required number of elements from the BS output space.

        :param input_state:
            A state that would be at the input of the interferometer. It serves as a
            source of knowledge about particles number and the modes number, thus
            generally about the output states space.
        :param samples_number:
            The number of samples to return.

        :return:
            Uniformly sampled required number of output states.
        """
        possible_output_states: List[Tuple[int, ...]] = generate_possible_states(
            sum(input_state), len(input_state)
        )

        return [
            possible_output_states[randint(0, len(possible_output_states))]
            for _ in range(samples_number)
        ]
