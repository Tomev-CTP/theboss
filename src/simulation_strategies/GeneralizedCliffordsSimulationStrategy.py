from src.simulation_strategies.SimulationStrategy import SimulationStrategy
from numpy import ndarray, zeros
from math import factorial
from typing import List
from copy import copy


class GeneralizedCliffordsSimulationStrategy(SimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray):
        self.r_sample = []
        self.number_of_input_photons = 0
        self.pmfs = []  # Probability mass functions calculated along the way.
        self.interferometer_matrix = interferometer_matrix
        self.input_state = []
        self._labeled_states = dict()

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
        #self.r_sample.sort()
        #return self.r_sample

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
        self.r_sample = []

        while self.number_of_input_photons > len(self.r_sample):
            self._calculate_another_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_another_layer_of_pmfs(self):
        possible_output_states = self._generate_possible_output_states()
        
        for output_state in possible_output_states:
            output_probability = self._calculate_outputs_probability(output_state)


    def _generate_possible_output_states(self):
        current_output_state = [0 for _ in range(len(self.interferometer_matrix))]
        for r_i in self.r_sample:
            current_output_state[r_i] = current_output_state[r_i] + 1
        possible_output_states = []
        for i in range(len(current_output_state)):
            new_possible_output = copy(current_output_state)
            new_possible_output[i] = new_possible_output[i] + 1
            possible_output_states.append(new_possible_output)
        return possible_output_states

    def _calculate_outputs_probability(self, output_state: ndarray):
        effective_scattering_matrix = self._calculate_effective_scattering_matrix(output_state)

    def _calculate_effective_scattering_matrix(self, output_state) -> ndarray:
        number_of_columns = sum(self.current_input_state)
        effective_scattering_matrix = zeros(shape=(len(number_of_columns, number_of_columns)))
        helper_matrix = zeros(shape=(len(self.interferometer_matrix, number_of_columns)))
        next_column_index = 0

        for s_i in self.current_input_state:
            for i in range(s_i):
                helper_matrix[:, [next_column_index]] = self.interferometer_matrix[:, [s_i]]
                next_column_index += 1
        next_row_index = 0

        for t_i in output_state:
            for i in range(t_i):
                effective_scattering_matrix[next_row_index, :] = helper_matrix[t_i, :]

                next_row_index += 1
        return effective_scattering_matrix

    def _sample_from_latest_pmf(self):
        yield NotImplementedError


s = GeneralizedCliffordsSimulationStrategy([])
s.simulate([1, 4, 2, 0, 0, 0, 0, 0])
