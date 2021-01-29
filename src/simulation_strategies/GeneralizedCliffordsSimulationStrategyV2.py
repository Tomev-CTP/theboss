__author__ = 'Tomasz Rybotycki'

from collections import defaultdict
from copy import copy
from math import factorial
from typing import List

from numpy import array, delete, insert, ndarray, int64, float64, append
from numpy.random import random

from src.boson_sampling_utilities.Boson_Sampling_Utilities import ChinHuhPermanentCalculator, particle_state_to_modes_state
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class GeneralizedCliffordsSimulationStrategyV2(SimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray) -> None:
        self.r_sample = []
        self.number_of_input_photons = 0
        self.pmfs = dict()  # Probability mass functions calculated along the way. Keys should be current r as tuples.
        self.interferometer_matrix = interferometer_matrix
        self.input_state = array([], dtype=int64)
        self._labeled_states = defaultdict(list)
        self.possible_outputs = dict()
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

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

    def _get_sorted_possible_states(self) -> None:
        """
            Calculate and sort all the substates of the input. They will later be used to calculate output
            probabilities.

            :return: Dict of all the possible substates of input (including 0 and the state), labeled with number of
            particles in this state.
        """
        # Calculating all possible substates of the input
        possible_input_states = self._calculate_all_input_substates(self.input_state.copy())

        # Labeling them into dict where keys are being number of particles in the state.
        self._labeled_states = defaultdict(list)

        for state in possible_input_states:
            states_particles_number = sum(state)
            self._labeled_states[states_particles_number].append(state)

    def _calculate_all_input_substates(self, state_part_left: ndarray) -> List[ndarray]:
        """
        Calculates substates of the input in recursive manner.

        :param state_part_left: State with reduced modes number.
        :return: All the substates for starting number of modes.
        """
        if len(state_part_left) < 1:
            return [array([], dtype=int64)]

        n = state_part_left[0]
        state_part_left = delete(state_part_left, 0)

        smaller_substates = self._calculate_all_input_substates(state_part_left.copy())

        substates = []
        for i in range(int(n + 1)):
            for substate in smaller_substates:
                new_substate = substate.copy()
                new_substate = insert(new_substate, 0, i)
                substates.append(new_substate)
        return substates

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
        weights = self._calculate_weights_from_k_vectors(array(corresponding_k_vectors, dtype=float))
        # weights /= sum(weights)
        self.possible_outputs[self.current_key] = self._generate_possible_output_states()

        pmf = []

        for output in self.possible_outputs[self.current_key]:
            output = particle_state_to_modes_state(output, len(self.input_state))
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(possible_input_states[i], output)
                probability *= weights[i]
                pmf[-1] += probability

        self.pmfs[self.current_key] = pmf

    def _calculate_weights_from_k_vectors(self, corresponding_k_vectors: ndarray) -> ndarray:
        return array([self._calculate_multinomial_coefficient(vector)
                      for vector in corresponding_k_vectors], dtype=float64)

    @staticmethod
    def _calculate_multinomial_coefficient(vector: ndarray) -> int:
        """
            Calculates multinomial coefficient of the vector, as proposed in Oszmaniec, Brod 2018
            (above formula 39).
            :param vector: Vector of which multinomial coefficient will be calculated.
            :return: Multinomial coefficient of given vector.
        """
        particles_number = sum(vector)
        multinomial_coefficient = factorial(particles_number)
        for value in vector:
            multinomial_coefficient /= factorial(value)

        return multinomial_coefficient

    def _generate_possible_output_states(self) -> List[ndarray]:
        possible_output_states = []

        for i in range(len(self.interferometer_matrix)):
            new_possible_output = copy(self.r_sample)
            new_possible_output = append(new_possible_output, [i])
            possible_output_states.append(array(new_possible_output, dtype=int64))

        return possible_output_states

    def _calculate_outputs_probability(self, input_state: ndarray, output_state: ndarray) -> float:
        permanent_calculator = ChinHuhPermanentCalculator(self.interferometer_matrix, input_state=input_state,
                                                          output_state=output_state)
        probability = abs(permanent_calculator.calculate()) ** 2

        for mode_occupation_number in input_state:
            probability /= factorial(mode_occupation_number)

        probability /= factorial(sum(input_state))

        return probability

    def _sample_from_latest_pmf(self) -> None:
        # sample_index = choice(arange(len(self.possible_outputs[self.current_key])), 1, p=self.pmfs[self.current_key])[0]  # Takes too long, due to casting.

        sample_index = 0
        random_value = random() * sum(self.pmfs[self.current_key]) # PMFs are not normalized.
        current_probability = 0
        for probability in self.pmfs[self.current_key]:
            current_probability += probability
            if current_probability > random_value:
                break
            sample_index += 1

        self.current_sample_probability = self.pmfs[self.current_key][sample_index]
        self.r_sample = self.possible_outputs[self.current_key][sample_index]
        self.current_key = tuple(self.r_sample)
