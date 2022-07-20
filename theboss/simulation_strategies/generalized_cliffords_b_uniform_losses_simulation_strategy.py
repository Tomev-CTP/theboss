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

from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
    GeneralizedCliffordsBSimulationStrategy,
    BSPermanentCalculatorInterface,
    mode_occupation_to_mode_assignment,
)
from numpy import ndarray, array, int64, zeros_like
from typing import List
from scipy.special import binom
from numpy.random import random


class GeneralizedCliffordsBUniformLossesSimulationStrategy(
    GeneralizedCliffordsBSimulationStrategy
):
    """
    An implementation of a more general approach to the GCCB algorithm taking into
    account the possible uniform losses. Notice that for transmissivity = 1 it
    works like GCCB algorithm.

    TODO TR:    Write tests for this method.

    TODO TR:    The permanent calculator should not be necessary for GCCB algorithms, as
                they use something the submatrices calculator and .
    """
    def __init__(
        self,
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        transmissivity: float = 1.0,
    ) -> None:
        super().__init__(bs_permanent_calculator)
        self._current_input = []
        self._working_input_state = None
        self._transmissivity: float = transmissivity
        self._binomial_weights: List[float] = []
        self._number_of_particles_in_sample: int = 0

    def _compute_particle_numbers_probabilities(self) -> None:
        """
        In BS with uniform losses the probabilities of obtaining a specific number of
        bosons in the output are given by the binomial weights. This method computes
        these weights for the use during sampling.
        """
        self._binomial_weights = []

        # Shorthand notation.
        n = self.number_of_input_photons
        eta = self._transmissivity

        for l in range(n + 1):
            self._binomial_weights.append(
                binom(n, l) * pow(eta, l) * pow(1 - eta, n - l)
            )

    def _get_number_of_particles_left(self) -> int:
        """
        Samples remaining particles number using binomial weights computed earlier.

        :return: Number of remaining particles.
        """
        number_of_particles_left: int = 0
        weights_sum = 0
        threshold = random()

        for i in range(len(self._binomial_weights)):
            weights_sum += self._binomial_weights[i]

            if weights_sum > threshold:
                return number_of_particles_left

            number_of_particles_left += 1

        return number_of_particles_left

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        """
        Returns sample from linear optics experiments given output state.

        :param input_state: Input state in particle basis.
        :param samples_number: Number of samples to simulate.
        :return: A resultant state after traversing through interferometer.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self._compute_particle_numbers_probabilities()

        particle_input_state = list(mode_occupation_to_mode_assignment(input_state))

        samples = []

        while len(samples) < samples_number:
            self._number_of_particles_in_sample = self._get_number_of_particles_left()
            self._current_input = zeros_like(input_state)
            self._working_input_state = particle_input_state.copy()
            self._fill_r_sample()
            samples.append(array(self.r_sample, dtype=int64))
        return samples

    def _fill_r_sample(self) -> None:
        """
        Fills a new sample. Notice that it trims the number of particles in the sample
        depending on a previously sampled number of particles in the output state.
        """
        self.r_sample = [0 for _ in self.input_state]

        while self._number_of_particles_in_sample > sum(self.r_sample):
            self._update_current_input()
            self._compute_pmf()
            self._sample_from_pmf()
