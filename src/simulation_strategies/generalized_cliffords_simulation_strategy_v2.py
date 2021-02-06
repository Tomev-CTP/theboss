__author__ = "Tomasz Rybotycki"

from copy import copy
from typing import List

from numpy import append, array, int64, ndarray

from src.boson_sampling_utilities.boson_sampling_utilities import particle_state_to_modes_state
from src.boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import \
    BSPermanentCalculatorInterface
from src.simulation_strategies.generalized_cliffords_simulation_strategy import GeneralizedCliffordsSimulationStrategy


class GeneralizedCliffordsSimulationStrategyInterfaceV2(GeneralizedCliffordsSimulationStrategy):
    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        super().__init__(bs_permanent_calculator)

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        """
            Returns sample from linear optics experiments given output state.

            :param input_state: Input state in particle basis.
            :param samples_number: Number of samples to simulate.
            :return: A resultant state after traversing through interferometer.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self._get_sorted_possible_states()
        self.pmfs = dict()

        samples = []

        while len(samples) < samples_number:
            self._fill_r_sample()
            samples.append(particle_state_to_modes_state(array(self.r_sample, dtype=int64), len(self.input_state)))
        return samples

    def _fill_r_sample(self) -> None:
        self.r_sample = []
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

        while self.number_of_input_photons > len(self.r_sample):
            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_new_layer_of_pmfs(self) -> None:
        number_of_particle_to_sample = len(self.r_sample) + 1
        possible_input_states = self._labeled_states[number_of_particle_to_sample]
        corresponding_k_vectors = [[self.input_state[i] - state[i] for i in range(len(state))]
                                   for state in possible_input_states]

        pmf = []

        weights = self._calculate_weights_from_k_vectors(array(corresponding_k_vectors, dtype=float))
        weights /= sum(weights)
        self.possible_outputs[self.current_key] = self._generate_possible_output_states()

        for output in self.possible_outputs[self.current_key]:
            output = particle_state_to_modes_state(output, len(self.input_state))
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(possible_input_states[i], output)
                probability *= weights[i]
                pmf[-1] += probability

        self.pmfs[self.current_key] = pmf

    def _generate_possible_output_states(self) -> List[ndarray]:
        possible_output_states = []

        for i in range(len(self.input_state)):
            new_possible_output = copy(self.r_sample)
            new_possible_output = append(new_possible_output, [i])
            possible_output_states.append(array(new_possible_output, dtype=int64))

        return possible_output_states
