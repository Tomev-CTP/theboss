__author__ = "Tomasz Rybotycki"

from random import random
from typing import List, Optional, Tuple, Sequence

from numpy import conjugate, exp, ndarray, ones, sqrt, zeros, pi
from numpy.random import rand

from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)
from theboss.network_simulation_strategy.lossy_network_simulation_strategy import (
    LossyNetworkSimulationStrategy,
)
from theboss.network_simulation_strategy.network_simulation_strategy import (
    NetworkSimulationStrategy,
)


class MeanFieldFixedLossSimulationStrategy(SimulationStrategyInterface):
    """
    A class implementing the mean-field simulation strategy for fixed losses.
    """

    def __init__(
        self,
        interferometer_matrix: Sequence[Sequence[complex]],
        number_of_photons_left: int,
        number_of_observed_modes: int,
        network_simulation_strategy: Optional[NetworkSimulationStrategy] = None,
    ) -> None:
        if network_simulation_strategy is None:
            network_simulation_strategy = LossyNetworkSimulationStrategy(
                interferometer_matrix
            )
        self.number_of_photons_left: int = number_of_photons_left
        self.interferometer_matrix: Sequence[Sequence[complex]] = interferometer_matrix
        self.number_of_observed_modes: int = number_of_observed_modes
        self._network_simulation_strategy = network_simulation_strategy

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Returns a sample from the mean-field distribution in fixed losses regime.

        :param samples_number:
            Number of samples one wants to simulate.
        :param input_state:
            Usually :math:`n`-particle Fock state with :math:`m` modes.

        :return:
            A list of samples from the mean-field distribution with fixed losses.
        """
        samples: List[Tuple[int, ...]] = []
        while len(samples) < samples_number:
            phi_0 = self._prepare_initial_state(input_state)
            evolved_state = self._network_simulation_strategy.simulate(
                input_state=phi_0
            )
            probabilities = self._calculate_probabilities(evolved_state)
            samples.append(
                tuple(
                    self._calculate_approximation_of_boson_sampling_outcome(
                        probabilities
                    )
                )
            )
        return samples

    def _prepare_initial_state(self, input_state: Sequence[int]) -> ndarray:
        """
        This method is used to prepare math:`\\psi_0` state (formula 23 from ref. [1]).

        :param input_state:
            Initial (lossy) Fock state.

        :return:
            Returns the initial state of the formula, which is an equal superposition
            of :math:`n` photons 'smeared' on the first :math:`n` modes.
        """
        initial_number_of_photons = int(sum(input_state))
        prepared_state = ones(self.number_of_observed_modes, dtype=float)
        prepared_state[initial_number_of_photons:] = 0
        prepared_state /= sqrt(
            initial_number_of_photons
        )  # Note, that numpy version of sqrt is used here!

        return self._randomize_modes_phases(prepared_state)

    @staticmethod
    def _randomize_modes_phases(state_in_modes_basis: ndarray) -> ndarray:
        """
        Randomize the phases of given mode state. Each mode should have different
        iid random phase.

        :param state_in_modes_basis:
            A given state in modes basis.

        :return:
            Given mode state with randomized phases.
        """
        return exp(1j * 2 * pi * rand(len(state_in_modes_basis))) * state_in_modes_basis

    @staticmethod
    def _calculate_probabilities(evolved_state: ndarray) -> ndarray:
        """
        Computes the probabilities of finding a particle in every mode.

        :param evolved_state:
            The approximate state evolved through the interferometer.

        :return:
            Probabilities of finding a particle in each of the modes.
        """
        return conjugate(evolved_state) * evolved_state

    def _calculate_approximation_of_boson_sampling_outcome(
        self, probabilities: ndarray
    ) -> ndarray:
        """
        This method applies evolution to every photon. Note, that evolution of each
        particle is independent of each other.

        :param probabilities:
            Probabilities of finding a particle in each of the modes.

        :return:
            A lossy boson state after traversing through interferometer. The state is
            described in first quantization (mode assignment representation).
        """
        output = zeros(self.number_of_observed_modes, dtype=int)
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
