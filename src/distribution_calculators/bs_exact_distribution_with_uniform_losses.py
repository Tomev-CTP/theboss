__author__ = "Tomasz Rybotycki"

from copy import deepcopy
from typing import List, Iterable

from numpy import ndarray
from scipy import special

from ..boson_sampling_utilities.boson_sampling_utilities import generate_possible_outputs
from ..distribution_calculators.bs_distribution_calculator_interface import \
    BosonSamplingExperimentConfiguration
from ..distribution_calculators.bs_distribution_calculator_with_fixed_losses import \
    BSDistributionCalculatorWithFixedLosses, BSPermanentCalculatorInterface


class BSDistributionCalculatorWithUniformLosses \
            (BSDistributionCalculatorWithFixedLosses):
    def __init__(self, configuration: BosonSamplingExperimentConfiguration,
                 permanent_calculator: BSPermanentCalculatorInterface) -> None:
        super().__init__(configuration, permanent_calculator)
        self.weights = self._initialize_weights()

    def _initialize_weights(self) -> List[float]:

        weight = \
            lambda n, l, eta: pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
        weights = []

        for number_of_particles_left \
                in range(self.configuration.initial_number_of_particles + 1):
            weights.append(
                weight(self.configuration.initial_number_of_particles,
                       number_of_particles_left,
                       self.configuration.uniform_transmissivity)
            )

        return weights

    def calculate_probabilities_of_outcomes(self,
                                            outcomes: Iterable[Iterable[int]]) -> \
    List[float]:
        outcomes_probabilities = [self.__calculate_probability_of_outcome(outcome) for
                                  outcome in outcomes]

        return outcomes_probabilities

    def __calculate_probability_of_outcome(self, outcome: ndarray) -> float:

        number_of_particles_left = int(sum(outcome))
        l = number_of_particles_left

        if l == 0:
            return self.weights[0]

        n = self.configuration.initial_number_of_particles

        subconfiguration = deepcopy(self.configuration)

        subconfiguration.number_of_particles_left = number_of_particles_left
        subconfiguration.number_of_particles_lost = n - l
        subdistribution_calculator = \
            BSDistributionCalculatorWithFixedLosses(subconfiguration,
                                                    self._permanent_calculator)

        probability_of_outcome = subdistribution_calculator.calculate_probabilities_of_outcomes([outcome])[0]

        return probability_of_outcome * self.weights[l]


    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        return generate_possible_outputs(self.configuration.initial_number_of_particles,
                                         self.configuration.number_of_modes, consider_loses=True)
