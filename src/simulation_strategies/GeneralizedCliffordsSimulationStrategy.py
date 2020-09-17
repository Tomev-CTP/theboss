__author__ = 'Tomasz Rybotycki'

from src.simulation_strategies.SimulationStrategy import SimulationStrategy
from numpy import ndarray, zeros, array
from numpy.random import choice
from numpy.linalg import norm
from math import factorial
from typing import List
from copy import copy
from src.Boson_Sampling_Utilities import calculate_permanent


class GeneralizedCliffordsSimulationStrategy(SimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray):
        self.r_sample = []
        self.number_of_input_photons = 0
        self.pmfs = []  # Probability mass functions calculated along the way.
        self.interferometer_matrix = interferometer_matrix
        self.input_state = []
        self._labeled_states = dict()
        self.current_outputs = []

    def simulate(self, input_state: ndarray) -> List[int]:
        """
            Returns sample from linear optics experiments given output state.
            :param input_state: Input state in particle basis.
            :return: A resultant state after traversing through interferometer.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self.__get_sorted_possible_states()
        self.__fill_r_sample()
        #print(self.r_sample)
        return self.r_sample

    def __get_sorted_possible_states(self) -> None:
        """
            Calculate and sort all the substates of the input. They will later be used to calculate output probabilities.
            :return: Dict of all the possible substates of input (including 0 and the state), labeled with number of
            particles in this state.
        """
        # Calculating all possible substates of the input
        possible_input_states = self.__calculate_all_input_substates(self.input_state[:])

        # Labeling them into dict where keys are being number of particles in the state.
        self._labeled_states = dict()
        particles_number = sum(self.input_state)

        for i in range(particles_number + 1):
            self._labeled_states[i] = []

        for state in possible_input_states:
            states_particles_number = sum(state)
            self._labeled_states[states_particles_number].append(state)

    def __calculate_all_input_substates(self, state_part_left: ndarray):
        """
        Calculates substates of the input in recursive manner.
        :param state_part_left: State with reduced modes number.
        :return: All the substates for starting number of modes.
        """
        if len(state_part_left) < 1:
            return [[]]
        n = state_part_left.pop(0)
        smaller_substates = self.__calculate_all_input_substates(state_part_left)
        substates = []
        for i in range(n + 1):
            for substate in smaller_substates:
                new_substate = substate[:]
                new_substate.insert(0, i)
                substates.append(new_substate)
        return substates

    def __fill_r_sample(self) -> None:
        self.pmfs = []
        self.r_sample = [0 for _ in range(len(self.interferometer_matrix))]

        while self.number_of_input_photons > sum(self.r_sample):
            self._calculate_another_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_another_layer_of_pmfs(self):
        number_of_particle_to_sample = sum(self.r_sample) + 1
        possible_input_states = self._labeled_states[number_of_particle_to_sample]
        corresponding_k_vectors = []
        for state in possible_input_states:
            corresponding_k_vectors.append([self.input_state[i] - state[i] for i in range(len(state))])
        weights = self._calculate_weights_from_k_vectors(corresponding_k_vectors)
        normalized_weights = weights / norm(weights)
        self.current_outputs = self._generate_possible_output_states()

        pmf = []

        '''
        print(f'r = {self.r_sample}')
        print('Outputs:')
        for output in self.current_outputs:
            print(f'\t {output}')
        print('Inputs: ')
        for input in possible_input_states:
            print(f'\t {input}')
        '''

        for output in self.current_outputs:
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(possible_input_states[i], output)
                probability *= normalized_weights[i] ** 2
                pmf[-1] += probability

        pmf = pmf / sum(pmf)
        '''
        print(pmf)
        print(sum(pmf))
        '''
        self.pmfs.append(pmf)

        # for output_state in possible_output_states:
        #    output_probability = self._calculate_outputs_probability(output_state)

    def _calculate_weights_from_k_vectors(self, corresponding_k_vectors: ndarray) -> ndarray:
        return [self._calculate_multinomial_coefficient(vector) for vector in corresponding_k_vectors]

    @staticmethod
    def _calculate_multinomial_coefficient(vector: ndarray):
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

    def _generate_possible_output_states(self):
        possible_output_states = []
        for i in range(len(self.r_sample)):
            new_possible_output = copy(self.r_sample)
            new_possible_output[i] = new_possible_output[i] + 1
            possible_output_states.append(new_possible_output)
        return possible_output_states

    def _calculate_outputs_probability(self, input_state: ndarray, output_state: ndarray):
        effective_scattering_matrix = self._calculate_effective_scattering_matrix(input_state, output_state)
        probability = calculate_permanent(effective_scattering_matrix) ** 2
        for mode_occupation_number in input_state:
            probability /= factorial(mode_occupation_number)
        for mode_occupation_number in output_state:
            probability /= factorial(mode_occupation_number)
        return probability

    def _calculate_effective_scattering_matrix(self, input_state: ndarray, output_state: ndarray) -> ndarray:
        number_of_columns = sum(input_state)
        effective_scattering_matrix = zeros(shape=(number_of_columns, number_of_columns))
        helper_matrix = zeros(shape=(len(self.interferometer_matrix), number_of_columns))
        next_column_index = 0

        for j in range(len(input_state)):
            for i in range(input_state[j]):
                helper_matrix[:, [next_column_index]] = self.interferometer_matrix[:, [j]]
                next_column_index += 1
        next_row_index = 0

        for j in range(len(output_state)):
            for i in range(int(output_state[j])):
                effective_scattering_matrix[[next_row_index], :] = helper_matrix[[j], :]

                next_row_index += 1

        return effective_scattering_matrix

    def _sample_from_latest_pmf(self):
        sample_index = choice([i for i in range(len(self.current_outputs))], 1, p=self.pmfs[-1])[0]
        self.r_sample = self.current_outputs[sample_index]
