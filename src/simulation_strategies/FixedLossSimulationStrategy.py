__author__ = "Tomasz Rybotycki"

from random import random
from typing import List, Optional

from numpy import conjugate, exp, ndarray, ones, sqrt, zeros
from numpy.linalg import norm
from numpy.random import rand

from src.network_simulation_strategy.LossyNetworkSimulationStrategy import LossyNetworkSimulationStrategy
from src.network_simulation_strategy.NetworkSimulationStrategy import NetworkSimulationStrategy
from src.simulation_strategies.SimulationStrategyInterface import SimulationStrategyInterface


class FixedLossSimulationStrategy(SimulationStrategyInterface):

    def __init__(self, interferometer_matrix: ndarray,
                 number_of_photons_left: int, number_of_observed_modes: int,
                 network_simulation_strategy: Optional[NetworkSimulationStrategy] = None) \
            -> None:
        if network_simulation_strategy is None:
            network_simulation_strategy = LossyNetworkSimulationStrategy(interferometer_matrix)
        self.number_of_photons_left = number_of_photons_left
        self.interferometer_matrix = interferometer_matrix
        self.number_of_observed_modes = number_of_observed_modes
        self._network_simulation_strategy = network_simulation_strategy

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        """
            Returns an sample from the approximate distribution in fixed losses regime.

            :param samples_number: Number of samples one wants to simulate.
            :param input_state: Usually n-particle Fock state in m modes.
            :return: A sample from the approximation.
        """
        samples = []
        while len(samples) < samples_number:
            phi_0 = self._prepare_initial_state(input_state)
            evolved_state = self._network_simulation_strategy.simulate(input_state=phi_0)
            probabilities = self._calculate_probabilities(evolved_state)
            samples.append(self._calculate_approximation_of_boson_sampling_outcome(probabilities))
        return samples

    def _prepare_initial_state(self, input_state: ndarray) -> ndarray:
        """
            This method is used to prepare psi_0 state (formula 23 from ref. [1]).
            :param input_state: Initial lossy bosonic state.
            :return: Returns the initial state of the formula, which is an equal superposition
            of n photons 'smeared' on the first n modes.
        """
        initial_number_of_photons = int(sum(input_state))
        prepared_state = ones(self.number_of_observed_modes, dtype=float)
        prepared_state[initial_number_of_photons:] = 0
        prepared_state /= sqrt(initial_number_of_photons)  # Note, that numpy version of sqrt is used here!

        return self._randomize_modes_phases(prepared_state)

    @staticmethod
    def _randomize_modes_phases(state_in_modes_basis: ndarray) -> ndarray:
        """
            Randomize the phases of given mode state. Each mode should have different iid random phase.
            :param state_in_modes_basis: A given state in modes basis.
            :return: Given mode state with randomized phases.
        """
        return exp(1j * rand(len(state_in_modes_basis))) * state_in_modes_basis

    @staticmethod
    def _calculate_probabilities(state: ndarray) -> ndarray:
        return conjugate(state) * state

    def _calculate_approximation_of_boson_sampling_outcome(self, probabilities: ndarray) -> ndarray:
        """
            This method applies evolution to every photon. Note, that evolution of each particle is independent of
            each other.
            :param probabilities:
            :return: A lossy boson state after traversing through interferometer. The state is described in first
            quantization (mode assignment basis).
        """
        output = zeros(self.number_of_observed_modes)
        for photon in range(self.number_of_photons_left):
            x = random()
            i = 0
            prob = probabilities[i]
            while x > prob:
                i += 1
                if len(probabilities) == i:
                    break
                prob += probabilities[i]
            if len(probabilities) != i:
                output[i] += 1
        return output
