__author__ = "Tomasz Rybotycki"

from copy import deepcopy
from typing import List

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

    def calculate_distribution(self) -> List[float]:
        """
        This method will be used to calculate the exact distribution of lossy boson sampling experiment.
        The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
        :return: List of probabilities of outcomes.
        """
        possible_outcomes = generate_possible_outputs(number_of_particles=self.configuration.initial_number_of_particles,
                                                      number_of_modes=self.configuration.number_of_modes,
                                                      consider_loses=True)
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
                BSDistributionCalculatorWithFixedLosses(subconfiguration, self._permanent_calculator)
            subdistribution = subdistribution_calculator.calculate_probabilities_of_outcomes([outcome for outcome in possible_outcomes if sum(outcome) == l])
            subdistribution_weight = pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
            subdistribution = [el * subdistribution_weight for el in subdistribution]

            exact_distribution.extend(subdistribution)

        return exact_distribution

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        return generate_possible_outputs(self.configuration.initial_number_of_particles,
                                         self.configuration.number_of_modes, consider_loses=True)
