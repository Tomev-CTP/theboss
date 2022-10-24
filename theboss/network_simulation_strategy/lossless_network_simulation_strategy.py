__author__ = "Tomasz Rybotycki"

from numpy import dot, ndarray
from typing import Sequence

from .network_simulation_strategy import NetworkSimulationStrategy


class LosslessNetworkSimulationStrategy(NetworkSimulationStrategy):
    """
    A class implementing the evolution of a state of distinguishable particles
    through an interferometer.
    """

    def __init__(self, matrix: Sequence[Sequence[complex]]) -> None:
        self._matrix: Sequence[Sequence[complex]] = matrix

    def simulate(self, input_state: Sequence[int]) -> Sequence[Sequence[complex]]:
        """
        Computes the dot product of matrix and the input state.

        :param input_state:
            The input state to be evolved.

        :return:
             The input state evolved through the interferometer matrix.
        """
        return dot(self._matrix, input_state)
