__author__ = 'Tomasz Rybotycki'

from numpy import ndarray, dot, sqrt, pi, exp, zeros, conjugate
from typing import List
from random import random
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class FixedLossSimulationStrategy(SimulationStrategy):

    def __init__(self, interferometer_matrix: ndarray, number_of_photons_left: int, number_of_observed_modes: int):
        self.number_of_photons_left = number_of_photons_left
        self.interferometer_matrix = interferometer_matrix
        self.number_of_observed_modes = number_of_observed_modes

    def simulate(self, input_state: ndarray) -> List[int]:
        """
        Returns an sample from the approximate distribution in fixed losses regime.
        :param input_state: Usually n-particle Fock state.
        :return: A sample from approximate.
        """
        phi_0 = self.__prepare_initial_state(input_state)
        evolved_state = dot(self.interferometer_matrix, phi_0)
        probabilities = self.__calculate_probabilities(evolved_state)
        return self.__calculate_approximation_of_boson_sampling_outcome(probabilities)

    def __prepare_initial_state(self, input_state: ndarray) -> ndarray:
        """
            This method is used to prepare psi_0 state (formula 23 from ref. [1]).
            :param input_state: Initial lossy bosonic state.
            :return:
        """
        initial_number_of_photons = sum(input_state)
        prepared_state = input_state[:]
        prepared_state /= sqrt(initial_number_of_photons)
        prepared_state = self.__randomize_modes_phases(prepared_state)
        return prepared_state

    @staticmethod
    def __randomize_modes_phases(state_in_modes_basis: ndarray) -> ndarray:
        """
            Randomize the phases of given mode state. Each mode should have different iid random
            :param state_in_modes_basis: A given state in modes basis.
            :return: Given mode state with randomized phases.
        """
        randomized_phases_state = []

        for mode in state_in_modes_basis:
            phi = random() * 2 * pi
            randomized_phases_state.append(exp(1j * phi) * mode)

        return randomized_phases_state

    @staticmethod
    def __calculate_probabilities(state: ndarray) -> ndarray:
        probabilities = []
        for detector in state:
            probabilities.append(conjugate(detector) * detector)
        return probabilities

    def __calculate_approximation_of_boson_sampling_outcome(self, probabilities: ndarray) -> ndarray:
        """
            This method applies evolution to every photon. Note, that evolution of each particle is independent of
            each other.
            :param probabilities:
            :return:
        """
        output = zeros(self.number_of_observed_modes)
        for photon in range(self.number_of_photons_left):
            x = random()
            i = 0
            prob = probabilities[i]
            while x > prob:
                i += 1
                prob += probabilities[i]
            output[i] += 1
        return output

