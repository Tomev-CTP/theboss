__author__ = "Tomasz Rybotycki"

from typing import List

from numpy import arange, ndarray
from numpy.random import choice
from scipy import special

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy
from src.simulation_strategies.SimulationStrategyInterface import SimulationStrategyInterface


class UniformLossSimulationStrategy(SimulationStrategyInterface):
    def __init__(self, interferometer_matrix: ndarray,
                 number_of_modes: int, transmissivity: float) \
            -> None:
        self.interferometer_matrix = interferometer_matrix
        self.number_of_modes = number_of_modes
        self.transmissivity = transmissivity

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        initial_number_of_particles = int(sum(input_state))

        # Using n, eta, l notation from the paper.
        n = initial_number_of_particles
        eta = self.transmissivity

        separable_states_weights = [pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l) for l in range(n + 1)]

        samples = []
        while len(samples) < samples_number:
            number_of_particles_left_in_selected_separable_state = choice(arange(0, n + 1), p=separable_states_weights)

            strategy = FixedLossSimulationStrategy(self.interferometer_matrix,
                                                   number_of_particles_left_in_selected_separable_state,
                                                   self.number_of_modes)

            simulator = BosonSamplingSimulator(strategy)

            samples.append(simulator.get_classical_simulation_results(input_state)[0])

        return samples
