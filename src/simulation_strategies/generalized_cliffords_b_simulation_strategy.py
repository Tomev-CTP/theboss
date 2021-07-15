__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of the Generalized Cliffords version B
    algorithm. It differs from the original version in two places:
        -   In the k-th step, instead of computing k x k permanents, we compute a set of 
            k-1 x k-1 permanents. From that, we calculate the required k x k permanents
            in O(k) time each.
        -   We permute the columns of the matrix at the beginning. We can then sample
            from an easier distribution and obtain the same results.
"""

from .generalized_cliffords_simulation_strategy import GeneralizedCliffordsSimulationStrategy, BSPermanentCalculatorInterface
from numpy import array, ndarray, int64, zeros_like
from typing import List
from numpy.random import choice

class GeneralizedCliffordsBSimulationStrategy(GeneralizedCliffordsSimulationStrategy):

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        super().__init__(bs_permanent_calculator)
        self._current_input = []
        self._working_input_state = None


    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        """
            Returns sample from linear optics experiments given output state.

            :param input_state: Input state in particle basis.
            :param samples_number: Number of samples to simulate.
            :return: A resultant state after traversing through interferometer.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self.pmfs = dict()

        samples = []

        while len(samples) < samples_number:
            self._current_input = zeros_like(input_state)
            self._working_input_state = list(input_state)
            self._fill_r_sample()
            samples.append(array(self.r_sample, dtype=int64))
        return samples

    def _calculate_new_layer_of_pmfs(self) -> None:

        self.possible_outputs[self.current_key] = self._generate_possible_output_states()

        pmf = []

        submatrices_permanents = self._get_permanents_of_submatrices()

        self._bs_permanent_calculator.input_state = self._current_input

        # New particle can come in any new mode
        for m in range(len(self.r_sample)):
            permanent = 0
            for i in range(len(self._current_input)):
                permanent_added = self._current_input[i] * submatrices_permanents[i]
                permanent_added *= self._bs_permanent_calculator.matrix[m][i]
                permanent += permanent_added

            pmf.append(abs(permanent)**2)

        self.pmfs[self.current_key] = pmf

    def _get_permanents_of_submatrices(self):
        permanents = []

        self._bs_permanent_calculator.output_state = self.r_sample

        for i in range(len(self._current_input)):

            if self._current_input[i] == 0:
                permanents.append(0)
                continue

            self._current_input[i] -= 1

            self._bs_permanent_calculator.input_state = self._current_input
            permanents.append(self._bs_permanent_calculator.compute_permanent())

            self._current_input[i] += 1

        return permanents

    def _fill_r_sample(self) -> None:
        self.pmfs.clear()
        self.r_sample = [0 for _ in self.input_state]
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

        while self.number_of_input_photons > sum(self.r_sample):
            self._update_current_input()

            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()

            self._sample_from_latest_pmf()

    def _update_current_input(self):
        next_particle_mode = choice(range(len(self._working_input_state)),
                                    p=self._working_input_state / sum(self._working_input_state))
        self._working_input_state[next_particle_mode] -= 1
        self._current_input[next_particle_mode] += 1
