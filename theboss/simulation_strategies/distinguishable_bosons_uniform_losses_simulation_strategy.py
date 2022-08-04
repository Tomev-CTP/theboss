__author__ = "Tomasz Rybotycki"

"""
    This script contains a class for simulating BS experiments with fully
    distinguishable bosons and uniform losses.
"""


from theboss.simulation_strategies.distinguishable_bosons_simulation_strategy import (
    DistinguishableBosonsSimulationStrategy,
)
from typing import Sequence, Tuple
from numpy.random import random


class DistinguishableBosonsUniformLossesSimulationStrategy(
    DistinguishableBosonsSimulationStrategy
):
    """
    A simulation strategy for lossless BS simulation with fully distinguishable
    bosons and uniform losses.
    """

    def __init__(self, matrix: Sequence[Sequence[complex]], transmissivity: float = 1):
        super().__init__(matrix)
        self._transmissivity: float = transmissivity

    def _get_new_sample(self, input_state: Sequence[int]) -> Tuple[int, ...]:
        """
        Generates a new sample for given input and set interferometer.

        :param input_state:
            A state at the input of the interferometer.

        :return:
            An output state from the BS experiment with distinguishable photons.
        """
        lossy_input: Tuple[int, ...] = self._apply_losses(input_state)
        return super()._get_new_sample(lossy_input)

    def _apply_losses(self, input_state: Sequence[int]) -> Tuple[int, ...]:
        """
        Applies uniform losses to the given input state.

        :param input_state:
            Input state to which losses will be applied.

        :return:
            Lossy input state.
        """
        lossy_input = [0 for _ in input_state]

        for mode in range(len(input_state)):
            for particle in range(input_state[mode]):
                if random() <= self._transmissivity:
                    lossy_input[mode] += 1

        return tuple(lossy_input)
