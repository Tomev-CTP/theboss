__author__ = "Tomasz Rybotycki"

from copy import deepcopy
from typing import List

from numpy import ndarray
from scipy import special

from src.boson_sampling_utilities.Boson_Sampling_Utilities import generate_possible_outputs
from src.distribution_calculators.BSDistributionCalculatorWithFixedLosses import \
    BSDistributionCalculatorWithFixedLosses, BSPermanentCalculatorInterface
from src.distribution_calculators.BSDistributionCalculatorInterface import \
    BosonSamplingExperimentConfiguration


class BosonSamplingWithUniformLossesExactDistributionCalculator \
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
                BSDistributionCalculatorWithFixedLosses(subconfiguration, self._permanent_calculator)
            possible_outcomes.extend(subdistribution_calculator.get_outcomes_in_proper_order())
            subdistribution = subdistribution_calculator.calculate_distribution()
            subdistribution_weight = pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
            subdistribution = [el * subdistribution_weight for el in subdistribution]

            exact_distribution.extend(subdistribution)

        return exact_distribution

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        return generate_possible_outputs(self.configuration.number_of_particles_left,
                                         self.configuration.number_of_modes, consider_loses=True)
