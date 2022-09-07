__author__ = "Tomasz Rybotycki"

from copy import deepcopy
from typing import List, Sequence, Tuple

from scipy import special

from theboss.boson_sampling_utilities import (
    generate_possible_states,
    compute_binomial_weights,
)
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
        weightless: bool = False,
    ) -> None:
        super().__init__(configuration, permanent_calculator)
        # self.weights = self._initialize_weights()
        self.weights = compute_binomial_weights(
            configuration.initial_number_of_particles,
            configuration.uniform_transmission_probability,
        )
        self.set_weightless(weightless)

    def set_weightless(self, weightless: bool) -> None:
        if weightless:
            self.weights = [1 for _ in self.weights]
        else:
            self.weights = compute_binomial_weights(
                self.configuration.initial_number_of_particles,
                self.configuration.uniform_transmission_probability,
            )
        self._weightless = weightless

    def calculate_probabilities_of_outcomes(
        self, outcomes: Sequence[Tuple[int, ...]]
    ) -> List[float]:

        with Pool(processes=cpu_count()) as pool:
            outcomes_probabilities = pool.map(
                self._calculate_probability_of_outcome, outcomes
            )

        return outcomes_probabilities

    def _calculate_probability_of_outcome(self, outcome: Tuple[int, ...]) -> float:

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
