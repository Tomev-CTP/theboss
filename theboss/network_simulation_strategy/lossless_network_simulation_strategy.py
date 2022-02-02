__author__ = "Tomasz Rybotycki"

from numpy import dot, ndarray

from .network_simulation_strategy import NetworkSimulationStrategy


class LosslessNetworkSimulationStrategy(NetworkSimulationStrategy):
    def __init__(self, matrix: ndarray) -> None:
        self._matrix = matrix

    def simulate(self, input_state: ndarray) -> ndarray:
        return dot(self._matrix, input_state)
