__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of the Generalized Cliffords version B
    algorithm. It differs from the original version  in two places:
        -   In the k-th step, instead of computing k x k permanents, we compute a set of 
            k-1 x k-1 permanents. From that, we calculate the required k x k permanents
            in O(k) time each.
        -   We permute the columns of the matrix at the beginning. We can then sample
            from an easier distribution and obtain the same results.
"""

from .generalized_cliffords_simulation_strategy import GeneralizedCliffordsSimulationStrategy, BSPermanentCalculatorInterface
from numpy import array, ndarray, int64
from typing import List
from numpy.random import choice

class GeneralizedCliffordsBSimulationStrategy(GeneralizedCliffordsSimulationStrategy):

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        self._current_input = []
        self._working_input_state = None
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
        self.pmfs = dict()

        samples = []

        while len(samples) < samples_number:
            self._current_input = [0 for _ in self.input_state]
            self._working_input_state = list(input_state)
            self._fill_r_sample()
            samples.append(array(self.r_sample, dtype=int64))
        return samples

    def _calculate_new_layer_of_pmfs(self) -> None:

        self.possible_outputs[self.current_key] = self._generate_possible_output_states()

        pmf = []

        self._bs_permanent_calculator.input_state = self._current_input

        for output in self.possible_outputs[self.current_key]:
            self._bs_permanent_calculator.output_state = output
            probability = abs(self._bs_permanent_calculator.compute_permanent())**2
            pmf.append(probability)

        self.pmfs[self.current_key] = pmf

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
