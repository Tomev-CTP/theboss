__author__ = "Tomasz Rybotycki"

from math import factorial
from numpy.random import random
from typing import List, Tuple, Sequence

from numpy import array_equal
from scipy import special

from .generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
)
from theboss.boson_sampling_utilities import generate_possible_states
from theboss.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)


class GeneralizedCliffordsUniformLossesSimulationStrategy(
    GeneralizedCliffordsSimulationStrategy
):
    def __init__(
        self,
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        transmission_probability: float = 0,
    ):
        self._transmission_probability = transmission_probability
        self.distribution = []
        self.unweighted_distribution = []
        self._possible_outputs = []
        self._binomial_weights = []
        self.missing_values_in_distribution = False
        super().__init__(bs_permanent_calculator)

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Returns sample from BS experiments given the input state.

        :param input_state:
            Input state in the mode occupation description.
        :param samples_number:
            Number of samples to simulate.

        :return:
            A resultant state after traversing through interferometer.
        """
        self._initialize_simulation(input_state)

        samples = []

        while len(samples) < samples_number:
            self._fill_r_sample()
            samples.append(tuple(self.r_sample))
        return samples

    def _initialize_simulation(self, input_state: Sequence[int]) -> None:
        """
        A method for algorithm initialization before each sampling.
        """

        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self._get_sorted_possible_states()
        self.pmfs = dict()

        distribution_initializer = 0

        if self.missing_values_in_distribution:
            distribution_initializer = -1  # -1 to indicate missing spots

        self._possible_outputs = generate_possible_states(
            sum(input_state), len(input_state), losses=True
        )
        self.distribution = [distribution_initializer for _ in self._possible_outputs]
        self.unweighted_distribution = [
            distribution_initializer for _ in self._possible_outputs
        ]

        n = sum(input_state)
        eta = self._transmission_probability

        # Do note that index is actually equal to number of particles left!
        self._binomial_weights = [
            pow(self._transmission_probability, left)
            * special.binom(n, left)
            * pow(1 - eta, n - left)
            for left in range(n + 1)
        ]
        self.distribution[0] = self._binomial_weights[0]

    def _fill_r_sample(self) -> None:
        """
        Fills the r_sample, but it's possible for the photons to be lost.
        """
        self.r_sample = [0 for _ in self._bs_permanent_calculator.matrix]
        self.current_key = tuple(self.r_sample)
        self.current_sample_probability = 1

        for i in range(self.number_of_input_photons):
            if random() >= self._transmission_probability:
                continue
            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_new_layer_of_pmfs(self) -> None:
        number_of_particle_to_sample = sum(self.r_sample) + 1
        possible_input_states = self._labeled_states[number_of_particle_to_sample]
        corresponding_k_vectors = [
            [self.input_state[i] - state[i] for i in range(len(state))]
            for state in possible_input_states
        ]

        pmf = []

        weights = self._compute_weights_from_k_vectors(corresponding_k_vectors)
        weights /= sum(weights)
        self.possible_outputs[
            self.current_key
        ] = self._generate_possible_output_states()

        for output in self.possible_outputs[self.current_key]:
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(
                    possible_input_states[i], output
                )
                probability *= weights[i]
                pmf[-1] += probability
            for i in range(len(self._possible_outputs)):
                if array_equal(output, self._possible_outputs[i]):
                    self.distribution[i] = pmf[-1]
                    self.distribution[i] *= factorial(sum(output))
                    for val in output:
                        self.distribution[i] /= factorial(val)
                    self.unweighted_distribution[i] = self.distribution[i]
                    self.distribution[i] *= self._binomial_weights[sum(output)]

        self.pmfs[self.current_key] = pmf
