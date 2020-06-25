__author__ = 'Tomasz Rybotycki'

import numpy as np
from math import factorial
from scipy.special import binom
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class BosonSamplingSimulator:

    def __init__(self, number_of_photons_left: int, initial_number_of_photons: int, number_of_observed_modes: int,
                 simulation_strategy: SimulationStrategy):
        self.number_of_photons_left = number_of_photons_left
        self.initial_number_of_photons = initial_number_of_photons
        self.number_of_observed_modes = number_of_observed_modes
        self.input_state = np.zeros(self.number_of_observed_modes)
        self.simulation_strategy = simulation_strategy

    def __prepare_input_state(self):
        """
            This method is used to prepare a general input state as a numpy array with size 1 x m, where
            m is the number of observed modes. The input state is the usual boson sampling input state
            (1, 1, ..., 1, 0, ..., 0), where there's exactly n ones, where n is the initial number of photons,
            according to Oszmaniec & Brod 2018.
        """
        self.input_state = np.zeros(self.number_of_observed_modes)
        self.input_state[:self.initial_number_of_photons] = 1

    def __modes_to_particles(self, state_in_modes_basis: list) -> list:
        """
            Transforms given Fock state into particle basis.
            :param state_in_modes_basis: A (Fock) state to transform into particle number basis.
            :return: Given state in particle number basis.
        """
        modes = np.zeros(self.number_of_observed_modes)
        modes[:] = state_in_modes_basis[:]
        state_in_particle_basis = np.zeros(self.number_of_photons_left)
        k = i = 0
        while i < self.number_of_observed_modes:
            if modes[i] > 0:
                modes[i] -= 1
                state_in_particle_basis[k] = int(i)
                k += 1
            else:
                i += 1
        return state_in_particle_basis

    def __particles_to_modes(self, state_in_particles_basis: list) -> list:
        """
            Transforms given state to modes basis.
            :param state_in_particles_basis:
            :return: A Fock state representing given particle basis state in modes basis.
        """
        modes = np.zeros(self.number_of_observed_modes)

        for particle in state_in_particles_basis:
            modes[int(particle)] += 1

        return modes

    @staticmethod
    def calculate_distance_from_lossy_bosonic_n_particle_state_to_set_of_symmetric_separable_l_particles_states(n, l):
        """
            Calculates the distance from lossy n-particles bosonic state, to the set of symmetric separable l-particles
            states. This is theorem 1 from ref. [1].

            :param n: Initial number of particles.
            :param l: Number of particles left.
            :return:
        """
        return 1. - factorial(n) / (n ** l * factorial(n - l))

    @staticmethod
    def calculate_number_of_outcomes_with_l_particles_in_m_modes(m, l):
        """
            Calculates number of possible l-particles outcomes in m modes.
            :param m: Number of observed modes.
            :param l: Number of particles left.
            :return: Number of l-particle outcomes in m modes.
        """
        # This has to be returned as int, because by default binom returns a float for some reason.
        return int(binom(m + l - 1, m - 1))

    def get_classical_simulation_results(self):
        self.__prepare_input_state()
        return self.simulation_strategy.simulate(self.input_state)
