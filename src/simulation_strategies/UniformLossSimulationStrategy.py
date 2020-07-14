__author__ = 'Tomasz Rybotycki'

from typing import List

from numpy import arange, ndarray
from numpy.random import choice
from scipy import special

from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class UniformLossSimulationStrategy(SimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray, number_of_modes: int, probability_of_uniform_loss: float) \
            -> None:
        self.interferometer_matrix = interferometer_matrix
        self.number_of_modes = number_of_modes
        self.probability_of_uniform_loss = probability_of_uniform_loss

    def simulate(self, input_state: ndarray) -> List[float]:
        initial_number_of_particles = int(sum(input_state))
        separable_states_weights = []

        # Using n, eta, l notation from the paper.
        n = initial_number_of_particles
        eta = self.probability_of_uniform_loss

        for number_of_particles_left in range(n + 1):
            l = number_of_particles_left
            separable_states_weights.append(pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l))

        number_of_particles_left_in_selected_separable_state = choice(arange(0, n + 1), p=separable_states_weights)

        strategy = FixedLossSimulationStrategy(self.interferometer_matrix,
                                               number_of_particles_left_in_selected_separable_state,
                                               self.number_of_modes)

        simulator = BosonSamplingSimulator(number_of_particles_left_in_selected_separable_state, n,
                                           self.number_of_modes, strategy)

        return simulator.get_classical_simulation_results()
