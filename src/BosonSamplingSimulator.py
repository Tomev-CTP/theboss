__author__ = 'Tomasz Rybotycki'

from typing import List

from numpy import zeros, ndarray
from scipy.special import binom

from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class BosonSamplingSimulator:

    def __init__(self, number_of_photons_left: int, initial_number_of_photons: int, number_of_observed_modes: int,
                 simulation_strategy: SimulationStrategy) -> None:
        self.number_of_photons_left = number_of_photons_left
        self.initial_number_of_photons = initial_number_of_photons
        self.number_of_observed_modes = number_of_observed_modes
        self.input_state = zeros(self.number_of_observed_modes)
        self.simulation_strategy = simulation_strategy

    @staticmethod
    def calculate_number_of_outcomes_with_l_particles_in_m_modes(m: int, l: int) -> int:
        """
            Calculates number of possible l-particles outcomes in m modes.
            :param m: Number of observed modes.
            :param l: Number of particles left.
            :return: Number of l-particle outcomes in m modes.
        """
        # This has to be returned as int, because by default binom returns a float for some reason.
        return int(binom(m + l - 1, m - 1))

    def get_classical_simulation_results(self, input_state: ndarray) -> List[int]:
        return self.simulation_strategy.simulate(input_state)
