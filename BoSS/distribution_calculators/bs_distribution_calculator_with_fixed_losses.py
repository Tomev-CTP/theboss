__author__ = "Tomasz Rybotycki"

import math
from copy import deepcopy
from typing import List, Iterable

from numpy import ndarray
from scipy import special

from ..boson_sampling_utilities.boson_sampling_utilities import generate_lossy_inputs, generate_possible_outputs
from ..boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import \
    BSPermanentCalculatorInterface
from ..distribution_calculators.bs_distribution_calculator_interface import BosonSamplingExperimentConfiguration, \
    BSDistributionCalculatorInterface


class BSDistributionCalculatorWithFixedLosses(BSDistributionCalculatorInterface):
    def __init__(self, configuration: BosonSamplingExperimentConfiguration,
                 permanent_calculator: BSPermanentCalculatorInterface) -> None:
        self.configuration = deepcopy(configuration)
        self._permanent_calculator = permanent_calculator

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        return generate_possible_outputs(self.configuration.number_of_particles_left,
                                         self.configuration.number_of_modes)

    def calculate_distribution(self) -> List[float]:
        """
        This method will be used to calculate the exact distribution of lossy boson sampling experiment.
        The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
        :return: List of probabilities of outcomes.
        """

        possible_outcomes = self.get_outcomes_in_proper_order()

        return self.calculate_probabilities_of_outcomes(possible_outcomes)

    def calculate_probabilities_of_outcomes(self, outcomes: Iterable[Iterable[int]] ) -> List[float]:
        outcomes_probabilities = [self.__calculate_probability_of_outcome(outcome) for
                                  outcome in outcomes]

        return outcomes_probabilities

    def __calculate_probability_of_outcome(self, outcome: ndarray) -> float:
        """
        Given input state and losses as described in the BS experiment configuration
        compute the probability of obtaining the outcome given as a parameter, after
        traversing a lossy linear-optical channel.
        :param outcome: A Fock state for which the probability will be computed.
        :return: Probability of obtaining given state in current experiment config.
        """
        outcome_probability = self.__calculate_probability_of_outcome_state_for_indistinguishable_photons(outcome)

        # Different states in particles-basis may give the same outcome state.
        outcome_probability *= math.factorial(self.configuration.number_of_particles_left)
        for i in range(self.configuration.number_of_modes):
            outcome_probability /= math.factorial(outcome[i])

        return outcome_probability

    def __calculate_probability_of_outcome_state_for_indistinguishable_photons(self, outcome_state: ndarray) -> float:
        probability_of_outcome = 0

        # Symmetrize the input.
        lossy_inputs_list = generate_lossy_inputs(self.configuration.initial_state,
                                                  self.configuration.number_of_particles_left)

        for lossy_input in lossy_inputs_list:
            self._permanent_calculator.matrix = self.configuration.interferometer_matrix
            self._permanent_calculator.input_state = lossy_input
            self._permanent_calculator.output_state = outcome_state
            subprobability = abs(self._permanent_calculator.compute_permanent()) ** 2
            for mode_occupation_number in lossy_input:
                subprobability /= math.factorial(mode_occupation_number)

            probability_of_outcome += subprobability

        # Normalization (read Brod & Oszmaniec 2019).
        probability_of_outcome /= math.factorial(self.configuration.number_of_particles_left)
        probability_of_outcome /= special.binom(self.configuration.initial_number_of_particles,
                                                self.configuration.number_of_particles_left)

        return probability_of_outcome
