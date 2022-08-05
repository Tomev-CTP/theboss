__author__ = "Tomasz Rybotycki"

from collections import defaultdict
from copy import copy
from math import factorial
from typing import List, Sequence, Tuple, DefaultDict
from scipy.special import binom

from numpy import array, delete, insert, int64, ndarray
from numpy.random import random

from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)
from theboss.permanent_calculators import BSPermanentCalculatorInterface


class GeneralizedCliffordsSimulationStrategy(SimulationStrategyInterface):
    """
    Generalized C&C strategy as proposed in [2].
    """

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        self.r_sample: List[int] = list()
        self.number_of_input_photons: int = 0
        self.pmfs = (
            dict()
        )  # Probability mass functions calculated along the way. Keys should be current r as tuples.
        self._bs_permanent_calculator = bs_permanent_calculator
        self.input_state: Sequence[int] = list()
        self._labeled_states: DefaultDict[int, List[Tuple[int, ...]]] = defaultdict(
            list
        )
        self.possible_outputs = dict()
        self.current_key: Tuple[int, ...] = tuple(self.r_sample)
        self.current_sample_probability: float = 1

    def set_new_matrix(self, new_matrix: Sequence[Sequence[complex]]) -> None:
        """
        Sets new interferometer matrix.

        :param new_matrix:
            New interferometer matrix.
        """
        self._bs_permanent_calculator.matrix = new_matrix

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Returns sample from linear optics experiments given output state.

        :param input_state:
            Input state in particle basis.
        :param samples_number:
            Number of samples to simulate.

        :return:
            A resultant state after traversing through interferometer.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self._get_sorted_possible_states()
        self.pmfs = dict()

        samples = []

        while len(samples) < samples_number:
            self._fill_r_sample()
            samples.append(tuple(self.r_sample))
        return samples

    def _get_sorted_possible_states(self) -> None:
        """
        Calculate and sort all the substates of the input. They will later be used
        to calculate output probabilities.

        :return:
            Dict of all the possible substates of input (including 0 and the state),
            labeled with number of particles in this state.
        """
        # Calculating all possible substates of the input
        possible_input_states = self._calculate_all_input_substates(
            copy(self.input_state)
        )

        # Labeling them into dict where keys are being number of particles in the state.
        self._labeled_states = defaultdict(list)

        for state in possible_input_states:
            self._labeled_states[sum(state)].append(state)

    def _calculate_all_input_substates(
        self, state_part_left: Sequence[int]
    ) -> List[ndarray]:
        """
        Calculates substates of the input in recursive manner.

        :param state_part_left:
            State with reduced modes number.

        :return:
            All the substates for starting number of modes.
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
        """
        Creates a sample according to the generalized C&C algorithm.
        """
        self.r_sample = [0 for _ in self.input_state]
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

        while self.number_of_input_photons > sum(self.r_sample):
            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_new_layer_of_pmfs(self) -> None:
        """
        Adds new layer, from which new particle will be sampled, to the pmfs dict.
        """
        number_of_particle_to_sample: int = sum(self.r_sample) + 1

        possible_input_states: List[Tuple[int, ...]] = self._labeled_states[
            number_of_particle_to_sample
        ]

        corresponding_k_vectors: List[List[int]] = [
            [self.input_state[i] - state[i] for i in range(len(state))]
            for state in possible_input_states
        ]

        weights: List[float] = self._compute_weights_from_k_vectors(
            corresponding_k_vectors
        )

        weights /= sum(weights)
        self.possible_outputs[
            self.current_key
        ] = self._generate_possible_output_states()

        pmf = []

        for output in self.possible_outputs[self.current_key]:
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(
                    possible_input_states[i], output
                )
                probability *= weights[i]
                pmf[-1] += probability

        self.pmfs[self.current_key] = pmf

    def _compute_weights_from_k_vectors(
        self, corresponding_k_vectors: List[List[int]]
    ) -> List[float]:
        """
        Computes the weights, as in [2], basing on the k_vectors.

        :param corresponding_k_vectors:
            A list of k_vectors from which the weights will be computed.
        :return:
        """
        return [self._calculate_weights(vector) for vector in corresponding_k_vectors]

    def _calculate_weights(self, k_vector: List[int]) -> float:
        """
        Computes the weight of the input state basing on the corresponding k_vector
        given as the argument.

        :param k_vector:
            The k_vector for which the weight will be computed.

        :return:
            The weight computed using the k_vector.

        """
        l: int = sum(k_vector)
        n: int = sum(self.input_state)

        weight: float = factorial(l) * factorial(n - l) / factorial(n)

        for m in range(len(self.input_state)):
            weight *= binom(self.input_state[m], k_vector[m])

        return weight

    def _generate_possible_output_states(self) -> List[List[int]]:
        """
        Generates a list of possible output states in the current step of the algorithm
        basing on the current r_sample.

        :return:
            A list of the output state that one may get in the current algorithm's step.
        """
        possible_output_states = []

        for i in range(len(self.r_sample)):
            new_possible_output = copy(self.r_sample)
            new_possible_output[i] += 1
            possible_output_states.append(new_possible_output)

        return possible_output_states

    def _calculate_outputs_probability(
        self, input_state: Sequence[int], output_state: Sequence[int]
    ) -> float:
        """
        Computes the probability of the output.

        :param input_state:
            Input state of the BS experiment instance.
        :param output_state:
            Output state of which the probability will be returned.

        :return:
            The probability of the given output state.
        """
        self._bs_permanent_calculator.input_state = input_state
        self._bs_permanent_calculator.output_state = output_state
        probability = abs(self._bs_permanent_calculator.compute_permanent()) ** 2

        for mode_occupation_number in input_state:
            probability /= factorial(mode_occupation_number)

        probability /= factorial(sum(input_state))

        return probability

    def _sample_from_latest_pmf(self) -> None:
        """
        Adds new sample to the output state and handles all the stuff that follows.
        """
        sample_index = 0
        random_value = random() * sum(
            self.pmfs[self.current_key]
        )  # PMFs are not normalized.
        current_probability = 0
        for probability in self.pmfs[self.current_key]:
            current_probability += probability
            if current_probability > random_value:
                break
            sample_index += 1

        self.current_sample_probability = self.pmfs[self.current_key][sample_index]
        self.r_sample = self.possible_outputs[self.current_key][sample_index]
        self.current_key = tuple(self.r_sample)
