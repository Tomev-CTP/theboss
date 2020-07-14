__author__ = 'Tomasz Rybotycki'

from numpy import zeros
from math import factorial
from scipy.special import binom
from typing import List
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class BosonSamplingSimulator:

    def __init__(self, number_of_photons_left: int, initial_number_of_photons: int, number_of_observed_modes: int,
                 simulation_strategy: SimulationStrategy) -> None:
        self.number_of_photons_left = number_of_photons_left
        self.initial_number_of_photons = initial_number_of_photons
        self.number_of_observed_modes = number_of_observed_modes
        self.input_state = zeros(self.number_of_observed_modes)
        self.simulation_strategy = simulation_strategy

    def __prepare_input_state(self) -> None:
        """
            This method is used to prepare a general input state as a numpy array with size 1 x m, where
            m is the number of observed modes. The input state is the usual boson sampling input state
            (1, 1, ..., 1, 0, ..., 0), where there's exactly n ones, where n is the initial number of photons,
            according to Oszmaniec & Brod 2018.
        """
        self.input_state = zeros(self.number_of_observed_modes)
        self.input_state[:self.initial_number_of_photons] = 1

    @staticmethod
    def calculate_distance_from_lossy_bosonic_n_particle_state_to_set_of_symmetric_separable_l_particles_states(n, l)\
            -> float:
        """
            Calculates the distance from lossy n-particles bosonic state, to the set of symmetric separable l-particles
            states. This is theorem 1 from ref. [1].

            :param n: Initial number of particles.
            :param l: Number of particles left.
            :return:
        """
        return 1. - factorial(n) / (n ** l * factorial(n - l))

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

    def get_classical_simulation_results(self) -> List[int]:
        self.__prepare_input_state()
        return self.simulation_strategy.simulate(self.input_state)
