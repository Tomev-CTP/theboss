__author__ = "Tomasz Rybotycki"

import math
from copy import deepcopy
from typing import List, Sequence, Tuple

from scipy.special import binom

from theboss.boson_sampling_utilities import (
    generate_lossy_n_particle_input_states,
    generate_possible_states,
)
from theboss.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)
from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BosonSamplingExperimentConfiguration,
    BSDistributionCalculatorInterface,
)


class BSDistributionCalculatorWithFixedLosses(BSDistributionCalculatorInterface):
    """
    This class contains the implementation of a BS distribution calculator for the
    regime where only a fixed number of particles is lost. It's a vital component of
    the BS distribution calculator with uniform losses.
    """

    def __init__(
        self,
        configuration: BosonSamplingExperimentConfiguration,
        permanent_calculator: BSPermanentCalculatorInterface,
    ) -> None:
        self.configuration = deepcopy(configuration)
        self._permanent_calculator = permanent_calculator

    def get_outcomes_in_proper_order(self) -> List[Tuple[int, ...]]:
        """
        Returns all possible outcomes of the BS experiment specified by the
        configuration. The "proper" order means only that in corresponds to the
        order of probabilities returned by the ``calculate_distribution`` method.

        :return:    All the possible outcomes of BS experiment specified by the
                    ``configuration``.
        """
        return generate_possible_states(
            self.configuration.number_of_particles_left,
            self.configuration.number_of_modes,
        )

    def calculate_distribution(self) -> List[float]:
        """
        This method will be used to calculate the exact distribution of (lossy)
        BosonSampling experiment. The results will be returned as a table of
        probabilities of obtaining the outcome at :math:`i`-th index.

        :return:
            List of probabilities of all possible outcomes.
        """
        return self.calculate_probabilities_of_outcomes(
            self.get_outcomes_in_proper_order()
        )

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
        return [self._calculate_probability_of_outcome(outcome) for outcome in outcomes]

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
        outcome_probability = (
            self._compute_probability_of_outcome_state_for_indistinguishable_photons(
                outcome
            )
        )

        # Different states in particles-basis may give the same outcome state.
        outcome_probability *= math.factorial(
            self.configuration.number_of_particles_left
        )
        for i in range(self.configuration.number_of_modes):
            outcome_probability /= math.factorial(outcome[i])

        return outcome_probability

    def _compute_probability_of_outcome_state_for_indistinguishable_photons(
        self, outcome_state: Tuple[int, ...]
    ) -> float:
        """
        Computes the probability of obtaining specified outcome state.

        :param outcome_state:
            A Fock state for which the probability will be computed.

        :return:
            The probability of obtaining outcome_state in a BS experiment for specified
            configuration.
        """
        probability_of_outcome = 0

        # Symmetrize the input.
        lossy_inputs_list = generate_lossy_n_particle_input_states(
            self.configuration.initial_state,
            self.configuration.number_of_particles_left,
        )

        for lossy_input in lossy_inputs_list:

            lossy_input_multiplicity = self._compute_lossy_input_multiplicity(
                lossy_input
            )

            self._permanent_calculator.matrix = self.configuration.interferometer_matrix
            self._permanent_calculator.input_state = lossy_input
            self._permanent_calculator.output_state = outcome_state

            subprobability = abs(self._permanent_calculator.compute_permanent()) ** 2

            for mode_occupation_number in lossy_input:
                subprobability /= math.factorial(mode_occupation_number)

            probability_of_outcome += subprobability * lossy_input_multiplicity

        # Normalization (read Brod & Oszmaniec 2019).
        probability_of_outcome /= math.factorial(sum(lossy_inputs_list[-1]))
        probability_of_outcome /= binom(
            self.configuration.initial_number_of_particles,
            self.configuration.number_of_particles_left,
        )

        return probability_of_outcome

    def _compute_lossy_input_multiplicity(self, lossy_input: Sequence[int]) -> int:
        """
        Computes the multiplicity of a lossy input. Some lossy states should be taken
        into account several times as you can "lose" a set of particles in a several
        ways that lead to the same lossy input in the end.

        :param lossy_input:
            The lossy input for which the multiplicity will be computed.

        :return:
            The multiplicity of a lossy state.
        """
        lossy_input_multiplicity = 1

        for i in range(len(lossy_input)):
            lossy_input_multiplicity *= binom(
                self.configuration.initial_state[i],
                self.configuration.initial_state[i] - lossy_input[i],
            )
        return int(lossy_input_multiplicity)
