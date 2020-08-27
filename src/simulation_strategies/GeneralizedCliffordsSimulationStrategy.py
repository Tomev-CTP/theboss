from src.simulation_strategies.SimulationStrategy import SimulationStrategy
from numpy import ndarray
from typing import List


class GeneralizedCliffordsSimulationStrategy(SimulationStrategy):
    def __init__(self):
        raise NotImplementedError

    def simulate(self, input_state: ndarray) -> List[int]:
        rSample = self._generateRSample()


    def _generateRSample(self):
        raise NotImplementedError
