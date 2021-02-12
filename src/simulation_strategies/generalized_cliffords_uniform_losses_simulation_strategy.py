__author__ = "Tomasz Rybotycki"

from math import factorial
from random import random
from typing import List

from numpy import array, int64, ndarray
from scipy import special

from src.boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import \
    BSPermanentCalculatorInterface
from src.distribution_calculators.bs_exact_distribution_with_uniform_losses import BosonSamplingExperimentConfiguration
from src.simulation_strategies.generalized_cliffords_simulation_strategy import \
    GeneralizedCliffordsSimulationStrategy

from src.boson_sampling_utilities.boson_sampling_utilities import generate_possible_outputs


class GeneralizedCliffordsUniformLossesSimulationStrategy(GeneralizedCliffordsSimulationStrategy):
    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface, transmissivity: float = 0):
        self._transmissivity = transmissivity
        self.distribution = []
        self._possible_outputs = []
        self._binomial_weights = []
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

        n = sum(input_state)
        eta = self._transmissivity

        self._possible_outputs = generate_possible_outputs(sum(input_state), len(input_state), consider_loses=True)
        self.distribution = [-1 for _ in self._possible_outputs] # -1 to indicate missing spots

        # Do note that index is actually equal to number of particles left!
        self._binomial_weights =\
            [pow(self._transmissivity, left) * special.binom(n, left) * pow(1 - eta, n - left) for left in range(n + 1)]
        self.distribution[0] = self._binomial_weights[0]

        samples = []

        while len(samples) < samples_number:
            self._fill_r_sample()
            samples.append(array(self.r_sample, dtype=int64))
        return samples

    def _fill_r_sample(self) -> None:
        """
            Fills the r_sample, but it's possible for the photons to be lost.
        """
        self.r_sample = [0 for _ in self._bs_permanent_calculator.matrix]
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

        for i in range(self.number_of_input_photons):
            if random() >= self._transmissivity:
                continue
            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_new_layer_of_pmfs(self) -> None:
        number_of_particle_to_sample = sum(self.r_sample) + 1
        possible_input_states = self._labeled_states[number_of_particle_to_sample]
        corresponding_k_vectors = [[self.input_state[i] - state[i] for i in range(len(state))]
                                   for state in possible_input_states]

        pmf = []

        weights = self._calculate_weights_from_k_vectors(array(corresponding_k_vectors, dtype=float))
        weights /= sum(weights)
        self.possible_outputs[self.current_key] = self._generate_possible_output_states()

        for output in self.possible_outputs[self.current_key]:
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(possible_input_states[i], output)
                probability *= weights[i]
                pmf[-1] += probability
            for i in range(len(self._possible_outputs)):
                if all(output == self._possible_outputs[i]):
                    self.distribution[i] = pmf[-1] * self._binomial_weights[sum(output)]
                    self.distribution[i] *= factorial(sum(output))
                    for val in output:
                        self.distribution[i] /= factorial(val)

        self.pmfs[self.current_key] = pmf
