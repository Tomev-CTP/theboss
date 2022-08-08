__author__ = "Tomasz Rybotycki"

"""
    This script contains an implementation of non-uniformly lossy BS distribution.
"""

from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BSDistributionCalculatorInterface,
)

from theboss.boson_sampling_utilities import (
    generate_possible_states,
    prepare_interferometer_matrix_in_expanded_space,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)

from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BSDistributionCalculatorWithUniformLosses,
    BosonSamplingExperimentConfiguration,
)

from typing import List, Sequence, Tuple, Set, Dict


class NonUniformlyLossyBSDistributionCalculator(BSDistributionCalculatorInterface):
    """
    A class that contains the calculator for computing non-uniformly lossy BS
    distribution.
    """

    def __init__(
        self,
        lossy_interferometer: Sequence[Sequence[complex]],
        input_state: Sequence[int],
    ):
        self._lossy_interferometer: Sequence[Sequence[complex]] = lossy_interferometer
        self._input_state: Sequence[int] = input_state

    @property
    def lossy_interferometer(self) -> Sequence[Sequence[complex]]:
        return self._lossy_interferometer

    @lossy_interferometer.setter
    def lossy_interferometer(
        self, lossy_interferometer: Sequence[Sequence[complex]]
    ) -> None:
        self._lossy_interferometer = lossy_interferometer

    @property
    def input_state(self) -> Sequence[int]:
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: Sequence[int]) -> None:
        self._input_state = input_state

    def calculate_probabilities_of_outcomes(
        self, outcomes: List[Sequence[int]]
    ) -> List[float]:
        """

        :param outcomes:
            Outputs for which the probability will be returned.

        :return:
            A list of probabilities of the outputs
        """
        probabilities: List[float] = []

        # Prepare experiment in the higher dimension.
        expanded_matrix: Sequence[
            Sequence[complex]
        ] = prepare_interferometer_matrix_in_expanded_space(self._lossy_interferometer)

        expanded_input: Tuple[int, ...] = tuple(self._input_state) + tuple(
            [0 for _ in self._input_state]
        )

        config: BosonSamplingExperimentConfiguration
        config = BosonSamplingExperimentConfiguration(
            expanded_matrix,
            expanded_input,
            sum(self._input_state),
            len(expanded_input),
            0,
            sum(self._input_state),
            1,
        )

        permanent_calculator: RyserPermanentCalculator = RyserPermanentCalculator(
            expanded_matrix, expanded_input, None
        )

        helper_distribution_calculator: BSDistributionCalculatorWithUniformLosses
        helper_distribution_calculator = BSDistributionCalculatorWithUniformLosses(
            config, permanent_calculator
        )

        required_expanded_outcomes: List[
            Tuple[int, ...]
        ] = self._generate_states_required_from_extended_dimension(outcomes)

        # Get expanded dimension output probabilities.
        expanded_outcomes_probabilities: List[
            float
        ] = helper_distribution_calculator.calculate_probabilities_of_outcomes(
            required_expanded_outcomes
        )

        # Get the marginals.
        for outcome in outcomes:
            outcome_probability: float = 0

            # Compute outcomes probability as marginal from expanded distribution.
            for j in range(len(required_expanded_outcomes)):
                if required_expanded_outcomes[j][0 : len(outcome)] == outcome:
                    outcome_probability += expanded_outcomes_probabilities[j]

            probabilities.append(outcome_probability)

        return probabilities

    def _generate_states_required_from_extended_dimension(
        self, outcomes: List[Sequence[int]]
    ) -> List[Tuple[int, ...]]:
        """
        Generates all the states required for computing the probabilities of the
        outcomes in the non-uniformly lossy BS experiment.

        :param outcomes:
            Outcomes for which the probabilities are being computed.

        :return:
            A list of outcomes from the extended dimension.
        """
        required_outcomes: List[Tuple[int, ...]] = []

        # Generate the "extensions"
        missing_particles_numbers: Set[int] = set()

        for outcome in outcomes:
            missing_particles_numbers.add(sum(self._input_state) - sum(outcome))

        outputs_extensions: Dict[int, List[Tuple[int, ...]]] = {}

        # Append proper extensions to each output state
        for outcome in outcomes:
            for extension in outputs_extensions[sum(self._input_state) - sum(outcome)]:
                required_outcomes.append(tuple(outcome) + extension)

        return required_outcomes

    def calculate_distribution(self) -> List[float]:
        """
        Computes and returns the whole non-uniformly lossy BS distribution. The order
        in which the probabilities are returned is given by get_outcomes_in_proper_order
        method.

        :return:
            The non-uniformly lossy BS outputs probability distribution.
        """
        return self.calculate_probabilities_of_outcomes(
            self.get_outcomes_in_proper_order()
        )

    def get_outcomes_in_proper_order(self) -> List[Tuple[int, ...]]:
        """
        Get the ordered output states, so that they correspond to the probabilities
        in the calculate_distribution method.

        :return:
            Ordered list of outcomes, where the order corresponds to the probabilities
            returned by the calculate_distribution method.
        """
        return generate_possible_states(
            sum(self._input_state), len(self._input_state), True
        )
