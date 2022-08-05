__author__ = "Tomasz Rybotycki"

"""
This script contains an implementation of a class for uniform sampling of the generally
lossy BS output states basing on the input states. It has been implemented for the tests
and validators.

Note: This is very naive implementation, as it only samples uniformly from all the
states.
"""

from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)

from typing import List, Tuple, Sequence
from theboss.boson_sampling_utilities import generate_possible_states
from numpy.random import randint


class UniformSamplingFromLossyOutputsStrategy(SimulationStrategyInterface):
    """
        A class for uniform sampling from the lossy BS outputs space.
    """

    def __init__(self) -> None:
        pass

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Draws required number of elements from the lossy BS output space.

        :param input_state:
            A state that would be at the input of the interferometer. It serves as a
            source of knowledge about particles number and the modes number, thus
            generally about the output states space.
        :param samples_number:
            The number of samples to return.

        :return:
            A list of output states sampled uniformly from lossy BS outputs.
        """
        possible_output_states: List[Tuple[int, ...]] = generate_possible_states(
            sum(input_state), len(input_state), True
        )

        return [
            possible_output_states[randint(0, len(possible_output_states))]
            for _ in range(samples_number)
        ]
