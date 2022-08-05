__author__ = "Tomasz Rybotycki"

from copy import deepcopy
from typing import List, Iterable, Tuple

from numpy import ndarray
from scipy import special

from theboss.boson_sampling_utilities import generate_possible_states
from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BosonSamplingExperimentConfiguration,
)
from ..distribution_calculators.bs_distribution_calculator_with_fixed_losses import (
    BSDistributionCalculatorWithFixedLosses,
    BSPermanentCalculatorInterface,
)
from multiprocessing import cpu_count, Pool


class BSDistributionCalculatorWithUniformLosses(
    BSDistributionCalculatorWithFixedLosses
):
    def __init__(
        self,
        configuration: BosonSamplingExperimentConfiguration,
        permanent_calculator: BSPermanentCalculatorInterface,
    ) -> None:
        super().__init__(configuration, permanent_calculator)
        self.weights = self._initialize_weights()
        self.weightless = False

    def set_weightless(self, weightless: bool) -> None:
        if not weightless:
            self.weights = self._initialize_weights()
        else:
            self.weights = [1 for _ in self.weights]

        self.weightless = weightless

    def _initialize_weights(self) -> List[float]:

        weight = (
            lambda n, l, eta: pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
        )
        weights = []

        for number_of_particles_left in range(
            self.configuration.initial_number_of_particles + 1
        ):
            weights.append(
                weight(
                    self.configuration.initial_number_of_particles,
                    number_of_particles_left,
                    self.configuration.uniform_transmissivity,
                )
            )

        return weights

    def calculate_probabilities_of_outcomes(
        self, outcomes: Iterable[Iterable[int]]
    ) -> List[float]:

        with Pool(processes=cpu_count()) as pool:
            outcomes_probabilities = pool.map(
                self._calculate_probability_of_outcome, outcomes
            )

        return outcomes_probabilities

    def _calculate_probability_of_outcome(self, outcome: ndarray) -> float:

        number_of_particles_left = int(sum(outcome))
        l = number_of_particles_left

        if l == 0:
            return self.weights[0]

        n = self.configuration.initial_number_of_particles

        subconfiguration = deepcopy(self.configuration)

        subconfiguration.number_of_particles_left = number_of_particles_left
        subconfiguration.number_of_particles_lost = n - l
        subdistribution_calculator = BSDistributionCalculatorWithFixedLosses(
            subconfiguration, self._permanent_calculator
        )

        probability_of_outcome = subdistribution_calculator.calculate_probabilities_of_outcomes(
            [outcome]
        )[
            0
        ]

        return probability_of_outcome * self.weights[l]

    def get_outcomes_in_proper_order(self) -> List[Tuple[int, ...]]:
        return generate_possible_states(
            self.configuration.initial_number_of_particles,
            self.configuration.number_of_modes,
            losses=True,
        )
