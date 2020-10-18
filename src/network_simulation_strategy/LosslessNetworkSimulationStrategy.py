__author__ = 'Tomasz Rybotycki'

from numpy import dot, ndarray

from src.network_simulation_strategy.NetworkSimulationStrategy import NetworkSimulationStrategy


class LossyNetworkSimulationStrategy(NetworkSimulationStrategy):
    def __init__(self, matrix: ndarray):
        self._matrix = matrix

    def simulate(self, input_state: ndarray) -> ndarray:
        return dot(self._matrix, input_state)
