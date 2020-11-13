__author__ = 'Tomasz Rybotycki'

from numpy import ndarray

from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class BosonSamplingSimulator:

    def __init__(self, simulation_strategy: SimulationStrategy) -> None:
        self.simulation_strategy = simulation_strategy

    def get_classical_simulation_results(self, input_state: ndarray) -> ndarray:
        return self.simulation_strategy.simulate(input_state)
