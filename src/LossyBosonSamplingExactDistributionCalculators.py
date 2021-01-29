__author__ = "Tomasz Rybotycki"

# TR TODO: This could be a part ob Boson_Sampling_Utilities package.

import abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from numpy import ndarray
from scipy import special

from src.boson_sampling_utilities.Boson_Sampling_Utilities import generate_lossy_inputs, generate_possible_outputs
from src.boson_sampling_utilities.permanent_calculators.BSPermanentCalculatorInterface import \
    BSPermanentCalculatorInterface
from src.network_simulation_strategy.NetworkSimulationStrategy import NetworkSimulationStrategy


# TODO TR: This class should be placed in separate folder
@dataclass
class BosonSamplingExperimentConfiguration:
    interferometer_matrix: ndarray  # A matrix describing interferometer.
    initial_state: ndarray
    initial_number_of_particles: int
    number_of_modes: int
    number_of_particles_lost: int
    number_of_particles_left: int
    uniform_transmissivity: float = 0
    network_simulation_strategy: NetworkSimulationStrategy = None
    lossy_modes_number: int = 0


class BosonSamplingExactDistributionCalculator(abc.ABC):
    """ Interface for boson sampling exact distribution calculators """

    @abc.abstractmethod
    def calculate_exact_distribution(self) -> List[float]:
        """ One has to be able to calculate exact distribution with it """
        raise NotImplementedError

    @abc.abstractmethod
    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        """ One also has to know the order of objects that returned probabilities correspond to """
        raise NotImplementedError


class BosonSamplingWithFixedLossesExactDistributionCalculator(BosonSamplingExactDistributionCalculator):
    def __init__(self, configuration: BosonSamplingExperimentConfiguration,
                 permanent_calculator: BSPermanentCalculatorInterface) -> None:
        self.configuration = deepcopy(configuration)
        self._permanent_calculator = permanent_calculator

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        return generate_possible_outputs(self.configuration.number_of_particles_left,
                                         self.configuration.number_of_modes)

    def calculate_exact_distribution(self) -> List[float]:
        """
        This method will be used to calculate the exact distribution of lossy boson sampling experiment.
        The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
        :return: List of probabilities of outcomes.
        """

        possible_outcomes = generate_possible_outputs(self.configuration.number_of_particles_left,
                                                      self.configuration.number_of_modes)

        outcomes_probabilities = [self.__calculate_probability_of_outcome(outcome) for outcome in possible_outcomes]

        return outcomes_probabilities

    def __calculate_probability_of_outcome(self, outcome: ndarray) -> float:
        """
        Given input state and losses as described in the constants of this file calculate the probability
        of obtaining the outcome given as a parameter, after traversing lossy linear-optical channel.
        :param outcome: An outcome which probability of obtaining will be calculated.
        :return: Probability of obtaining given outcome in situation presented by by the
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
            subprobability = abs(self._permanent_calculator.calculate()) ** 2
            for mode_occupation_number in lossy_input:
                subprobability /= math.factorial(mode_occupation_number)

            probability_of_outcome += subprobability

        # Normalization (read Brod & Oszmaniec 2019).
        probability_of_outcome /= math.factorial(self.configuration.number_of_particles_left)
        probability_of_outcome /= special.binom(self.configuration.initial_number_of_particles,
                                                self.configuration.number_of_particles_left)

        return probability_of_outcome


class BosonSamplingWithUniformLossesExactDistributionCalculator \
            (BosonSamplingWithFixedLossesExactDistributionCalculator):
    def __init__(self, configuration: BosonSamplingExperimentConfiguration,
                 permanent_calculator: BSPermanentCalculatorInterface) -> None:
        super().__init__(configuration, permanent_calculator)

    def calculate_exact_distribution(self) -> List[float]:
        """
        This method will be used to calculate the exact distribution of lossy boson sampling experiment.
        The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
        :return: List of probabilities of outcomes.
        """
        possible_outcomes = []
        exact_distribution = []

        # Using eta, n and l notation from the paper for readability purposes.
        n = self.configuration.initial_number_of_particles
        eta = self.configuration.uniform_transmissivity

        for number_of_particles_left in range(n + 1):  # +1 to include situation with all particles left.

            l = number_of_particles_left

            subconfiguration = deepcopy(self.configuration)

            subconfiguration.number_of_particles_left = number_of_particles_left
            subconfiguration.number_of_particles_lost = n - l
            subdistribution_calculator = \
                BosonSamplingWithFixedLossesExactDistributionCalculator(subconfiguration, self._permanent_calculator)
            possible_outcomes.extend(subdistribution_calculator.get_outcomes_in_proper_order())
            subdistribution = subdistribution_calculator.calculate_exact_distribution()
            subdistribution_weight = pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
            subdistribution = [el * subdistribution_weight for el in subdistribution]

            exact_distribution.extend(subdistribution)

        return exact_distribution

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        possible_outcomes = []

        for number_of_particles_left in range(self.configuration.initial_number_of_particles + 1):
            subconfiguration = deepcopy(self.configuration)
            subconfiguration.number_of_particles_left = number_of_particles_left
            subdistribution_calculator = \
                BosonSamplingWithFixedLossesExactDistributionCalculator(subconfiguration)
            possible_outcomes.extend(subdistribution_calculator.get_outcomes_in_proper_order())

        return possible_outcomes
