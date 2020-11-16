__author__ = 'Tomasz Rybotycki'

from collections import defaultdict
from copy import copy
from math import factorial
from typing import List

from numpy import arange, array, delete, insert, ndarray, int64, float64
from numpy.linalg import norm
from numpy.random import choice

from src.Boson_Sampling_Utilities import ChinHuhPermanentCalculator
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class GeneralizedCliffordsSimulationStrategy(SimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray) -> None:
        self.r_sample = []
        self.number_of_input_photons = 0
        self.pmfs = dict()  # Probability mass functions calculated along the way. Keys should be current r as tuples.
        self.interferometer_matrix = interferometer_matrix
        self.input_state = array([], dtype=int64)
        self._labeled_states = defaultdict(list)
        self.possible_outputs = dict()
        self.current_key = tuple(self.r_sample)

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        """
            Returns sample from linear optics experiments given output state.

            :param input_state: Input state in particle basis.
            :param samples_number: Number of samples to simulate.
            :return: A resultant state after traversing through interferometer.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self.__get_sorted_possible_states()
        self.pmfs = dict()

        samples = []

        while len(samples) < samples_number:
            self.__fill_r_sample()
            samples.append(array(self.r_sample, dtype=int64))
        return samples

    def __get_sorted_possible_states(self) -> None:
        """
            Calculate and sort all the substates of the input. They will later be used to calculate output
            probabilities.

            :return: Dict of all the possible substates of input (including 0 and the state), labeled with number of
            particles in this state.
        """
        # Calculating all possible substates of the input
        possible_input_states = self.__calculate_all_input_substates(self.input_state.copy())

        # Labeling them into dict where keys are being number of particles in the state.
        self._labeled_states = defaultdict(list)

        for state in possible_input_states:
            states_particles_number = sum(state)
            self._labeled_states[states_particles_number].append(state)

    def __calculate_all_input_substates(self, state_part_left: ndarray) -> List[ndarray]:
        """
        Calculates substates of the input in recursive manner.

        :param state_part_left: State with reduced modes number.
        :return: All the substates for starting number of modes.
        """
        if len(state_part_left) < 1:
            return [array([], dtype=int64)]

        n = state_part_left[0]
        state_part_left = delete(state_part_left, 0)

        smaller_substates = self.__calculate_all_input_substates(state_part_left.copy())

        substates = []
        for i in range(int(n + 1)):
            for substate in smaller_substates:
                new_substate = substate.copy()
                new_substate = insert(new_substate, 0, i)
                substates.append(new_substate)
        return substates

    def __fill_r_sample(self) -> None:
        self.r_sample = [0 for _ in self.interferometer_matrix]
        self.current_key = tuple(self.r_sample)

        while self.number_of_input_photons > sum(self.r_sample):
            if self.current_key not in self.pmfs:
                self.__calculate_new_layer_of_pmfs()
            self.__sample_from_latest_pmf()

    def __calculate_new_layer_of_pmfs(self) -> None:
        number_of_particle_to_sample = sum(self.r_sample) + 1
        possible_input_states = self._labeled_states[number_of_particle_to_sample]
        corresponding_k_vectors = [[self.input_state[i] - state[i] for i in range(len(state))]
                                   for state in possible_input_states]
        weights = self.__calculate_weights_from_k_vectors(array(corresponding_k_vectors, dtype=int64))
        normalized_weights = weights / norm(weights)
        self.possible_outputs[self.current_key] = self.__generate_possible_output_states()

        pmf = []

        for output in self.possible_outputs[self.current_key]:
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self.__calculate_outputs_probability(possible_input_states[i], output)
                probability *= normalized_weights[i] ** 2
                pmf[-1] += probability

        probabilities_sum = sum(pmf)

        pmf = [probability / probabilities_sum for probability in pmf]
        self.pmfs[self.current_key] = pmf

    def __calculate_weights_from_k_vectors(self, corresponding_k_vectors: ndarray) -> ndarray:
        return array([self.__calculate_multinomial_coefficient(vector)
                      for vector in corresponding_k_vectors], dtype=float64)

    @staticmethod
    def __calculate_multinomial_coefficient(vector: ndarray) -> int:
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

    def __generate_possible_output_states(self) -> List[ndarray]:
        possible_output_states = []
        for i in range(len(self.r_sample)):
            new_possible_output = copy(self.r_sample)
            new_possible_output[i] += 1
            possible_output_states.append(array(new_possible_output, dtype=int64))
        return possible_output_states

    def __calculate_outputs_probability(self, input_state: ndarray, output_state: ndarray) -> float:
        permanent_calculator = ChinHuhPermanentCalculator(self.interferometer_matrix, input_state=input_state,
                                                          output_state=output_state)
        probability = abs(permanent_calculator.calculate()) ** 2
        for mode_occupation_number in input_state:
            probability /= factorial(mode_occupation_number)
        for mode_occupation_number in output_state:
            probability /= factorial(mode_occupation_number)
        return probability

    def __sample_from_latest_pmf(self) -> None:
        sample_index = choice(arange(len(self.possible_outputs[self.current_key])), 1, p=self.pmfs[self.current_key])[0]
        self.r_sample = self.possible_outputs[self.current_key][sample_index]
        self.current_key = tuple(self.r_sample)
