__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of distribution calculator for special
    separable states described in [1] i [2]. For a given matrix U we would expect
    that the distribution returned would be _good_ approximation of U.
"""

from typing import Iterable, List

from numpy import ndarray, ones_like, sqrt, diag, zeros, hstack

from .bs_distribution_calculator_interface import (
    BSDistributionCalculatorInterface,
    BosonSamplingExperimentConfiguration,
)
from ..boson_sampling_utilities.boson_sampling_utilities import (
    generate_possible_states,
    generate_qft_matrix_for_first_m_modes,
    generate_random_phases_matrix_for_first_m_modes,
    prepare_interferometer_matrix_in_expanded_space,
)
from ..boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)

from .bs_distribution_calculator_with_fixed_losses import (
    BSDistributionCalculatorWithFixedLosses,
)


class BSApproximatedLossyDistributionCalculator(BSDistributionCalculatorInterface):
    def __init__(
        self,
        configuration: BosonSamplingExperimentConfiguration,
        permanent_calculator: BSPermanentCalculatorInterface,
    ) -> None:
        self._configuration = configuration
        self._permanent_calculator = permanent_calculator
        self._initial_matrix = self._prepare_initial_matrix()
        self._initial_state = self._prepare_initial_state()
        # How many matrices should I average over to get the distribution?
        # TODO TR: How many of these should I actually put here?
        self._matrices_number = 100

    def _prepare_initial_matrix(self):
        # TODO TR: This is basically the same as in nonuniform_losses_approximation_strategy.
        # I should do something about that.
        loss_removing_matrix = ones_like(self._configuration.interferometer_matrix[0])
        loss_removing_matrix[: self._approximated_modes_number] = 1.0 / sqrt(
            self._modes_transmissivity
        )  # This here assumes uniform losses
        loss_removing_matrix = diag(loss_removing_matrix)

        initial_matrix = (
            self._configuration.interferometer_matrix @ loss_removing_matrix
        )

        initial_matrix = prepare_interferometer_matrix_in_expanded_space(initial_matrix)

        return initial_matrix

    def _prepare_initial_state(self):
        """
            Initial state of approximate simulation (before applying qft and random
            phases) should have all the particles from approximated modes (we assume
            that we approximate first m modes) in the last approximated mode.

            Something like:
            [0, 0, ..., 0, k, NON-APPROXIMATED REST OF THE STATE]

            For simplicity I also extend it to 2m.
        """

        # Extend the state to 2m
        initial_state = zeros(2 * self._configuration.number_of_modes, dtype=int)
        initial_state[
            : self._configuration.number_of_modes
        ] = self._configuration.initial_state

        # Bunch approximated modes
        number_of_particles_to_put_in_the_last_approximated_mode = 0
        for i in range(self._configuration.approximated_modes_number - 1):
            number_of_particles_to_put_in_the_last_approximated_mode += initial_state[i]
            initial_state[i] = 0

        initial_state[
            self._configuration.approximated_modes_number - 1
        ] = number_of_particles_to_put_in_the_last_approximated_mode

        return initial_state

    def calculate_distribution(self) -> List[float]:
        """
            Computes whole distribution basing on configuration.
        """
        possible_outcomes = generate_possible_states(
            self.configuration.number_of_particles_left,
            self.configuration.number_of_modes,
        )
        return self.calculate_probabilities_of_outcomes(possible_outcomes)

    def calculate_probabilities_of_outcomes(
        self, outcomes: Iterable[Iterable[int]]
    ) -> List[float]:
        probabilities = []

        # TODO TR: We could probably make this parallel.
        for _ in range(self._matrices_number):
            effective_matrix = self._generate_effective_matrix()
            for outcome in outcomes:
                probabilities.append(
                    self._calculate_probability_of_outcome(outcome, effective_matrix)
                )

        return probabilities

    def _generate_effective_matrix(self):
        effective_matrix = generate_qft_matrix_for_first_m_modes(
            self._configuration.approximated_modes_number,
            self._configuration.number_of_modes * 2,
        )
        effective_matrix = (
            effective_matrix
            @ generate_random_phases_matrix_for_first_m_modes(
                self._configuration.approximated_modes_number,
                self._configuration.number_of_modes * 2,
            )
        )
        effective_matrix = effective_matrix @ self._initial_matrix
        return effective_matrix

    def _calculate_probability_of_outcome(
        self, outcome: Iterable[int], matrix: ndarray
    ) -> float:
        # First thing to note is that I can calculate probabilities of 2m-mode states,
        # but what I need is m-mode states' probabilities.

        # As a matter of fact I can compute 1m probability from 2m probability, by
        # summing probabilities of these states that have outcome state on first m modes
        # (first m, because we prepare the effective matrix in such a way).
        considered_outputs = self._get_2m_outcomes_corresponding_to_the_outcome(outcome)

        # I want to compute lossless distribution (or even probabilities of the lossless
        # outputs), so I may use one of the distribution calculators that I've got.
        subproblem_configuration = BosonSamplingExperimentConfiguration(
            interferometer_matrix=matrix,
            initial_state=self._initial_state,
            initial_number_of_particles=self._configuration.initial_number_of_particles,
            number_of_modes=2 * self._configuration.number_of_modes,
            number_of_particles_lost=0,
            number_of_particles_left=self._configuration.initial_number_of_particles,
        )
        subdistribution_calculator = BSDistributionCalculatorWithFixedLosses(
            permanent_calculator=self._permanent_calculator,
            configuration=subproblem_configuration,
        )
        probabilities_of_outcomes = subdistribution_calculator.calculate_probabilities_of_outcomes(
            outcomes=considered_outputs
        )
        return sum(probabilities_of_outcomes)

    def _get_2m_outcomes_corresponding_to_the_outcome(
        self, outcome: Iterable[int]
    ) -> List[Iterable[int]]:
        """
            This method returns 2m outcomes for given particle number, such that
            in the first m modes we've got exactly output state. Assuming that there
            are n particles total and k particles in the given outcome, we compute
            all m-mode (n-k)-particles states and just hstack them with output.
        """
        considered_outcomes = []
        possible_m_mode_outputs_with_less_particles = generate_possible_states(
            particles_number=self._configuration.initial_number_of_particles
                             - sum(outcome),
            modes_number=self._configuration.number_of_modes,
        )

        for possible_output in possible_m_mode_outputs_with_less_particles:
            considered_outcomes.append(hstack([outcome, possible_output]))

        return considered_outcomes

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        """
            Returns states in the same order that distribution probabilities were
            calculated in.
        """
        return generate_possible_states(
            self._configuration.number_of_particles_left,
            self._configuration.number_of_modes,
        )
