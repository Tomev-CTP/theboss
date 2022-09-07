__author__ = "Tomasz Rybotycki"

"""
This script contains an implementation of the distribution calculator for uniformly
lossy distinguishable BS experiments.
"""

from copy import deepcopy
from typing import List, Sequence, Tuple

from theboss.boson_sampling_utilities import (
    generate_possible_states,
    compute_binomial_weights,
)
from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BosonSamplingExperimentConfiguration,
)
from theboss.distribution_calculators.fixed_losses_distinguishable_particles_distribution_calculator import (
    FixedLossesDistinguishableParticlesDistributionCalculator,
    BSPermanentCalculatorInterface,
)
from multiprocessing import cpu_count, Pool


class UniformLossesDistinguishableParticlesDistributionCalculator(
    FixedLossesDistinguishableParticlesDistributionCalculator
):
    def __init__(
        self,
        configuration: BosonSamplingExperimentConfiguration,
        permanent_calculator: BSPermanentCalculatorInterface,
        weightless: bool = False,
    ) -> None:
        super().__init__(configuration, permanent_calculator)
        self.weights: List[float]
        self.set_weightless(weightless)

    def set_weightless(self, weightless: bool) -> None:
        """
        Sets the calculator to work in the weightless modes if necessary.

        :param weightless:
            A flag to inform the calculator in which mode it should work.
        """
        if weightless:
            self.weights = [1 for _ in self.weights]
        else:
            self.weights = compute_binomial_weights(
                sum(self.configuration.initial_state),
                self.configuration.uniform_transmission_probability,
            )

        self.weightless = weightless

    def calculate_probabilities_of_outcomes(
        self, outcomes: Sequence[Tuple[int, ...]]
    ) -> List[float]:
        """
        Computes and returns the probabilities of obtaining specified outcomes in the
        BS experiment described by the configuration. The order of the probabilities
        corresponds to that of the outcomes.

        :param outcomes:
            A list of Fock states for which the probabilities will be computed.

        :return:
            A list of probabilities of obtaining specified outcomes.
        """
        with Pool(processes=cpu_count()) as pool:
            outcomes_probabilities = pool.map(
                self._calculate_probability_of_outcome, outcomes
            )

        return outcomes_probabilities

    def _calculate_probability_of_outcome(self, outcome: Tuple[int, ...]) -> float:
        """
        Given input state and losses as described in the BS experiment configuration
        compute the probability of obtaining the outcome given as a parameter, after
        traversing a lossy linear-optical channel.

        :param outcome:
            A Fock state for which the probability will be computed.

        :return:
            Probability of obtaining given state in current experiment configuration.
        """
        number_of_particles_left = sum(outcome)
        l = number_of_particles_left

        if l == 0:
            return self.weights[0]

        n = self.configuration.initial_number_of_particles

        subconfiguration = deepcopy(self.configuration)

        subconfiguration.number_of_particles_left = number_of_particles_left
        subconfiguration.number_of_particles_lost = n - l

        subdistribution_calculator = FixedLossesDistinguishableParticlesDistributionCalculator(
            subconfiguration, self._permanent_calculator
        )

        probability_of_outcome = subdistribution_calculator.calculate_probabilities_of_outcomes(
            [outcome]
        )[
            0
        ]

        return probability_of_outcome * self.weights[l]

    def get_outcomes_in_proper_order(self) -> List[Tuple[int, ...]]:
        """
        Returns all possible outcomes of the BS experiment specified by the
        configuration. The "proper" order means only that in corresponds to the
        order of probabilities returned by the calculate_distribution method.

        :return:
            All the possible outcomes of BS experiment specified by the configuration.
        """
        return generate_possible_states(
            sum(self.configuration.initial_state),
            self.configuration.number_of_modes,
            losses=True,
        )
