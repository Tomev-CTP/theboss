__author__ = "Tomasz Rybotycki"

from typing import List, Sequence, Tuple, Dict

"""
    This script contains the implementation of the probabilities calculator for the 
    uniform sampler from the uniformly lossy BS distribution. 
"""

from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BSDistributionCalculatorInterface,
)
from theboss.boson_sampling_utilities import (
    bosonic_space_dimension,
    generate_possible_states,
    compute_binomial_weights,
)


class LosslessUniformBSDistributionCalculator(BSDistributionCalculatorInterface):
    """
    A class for computing the output probabilities of the uniformly lossy uniform
    sampling from BS distribution with a fixed number of initial particles and modes.
    """

    def __init__(
        self, modes_number: int, particles_number: int, transmissivity: float
    ) -> None:
        self._modes_number: int = modes_number
        self._particles_number: int = particles_number
        self._transmissivity: float = transmissivity

    @property
    def modes_number(self) -> int:
        return self._modes_number

    @modes_number.setter
    def modes_number(self, modes_number) -> None:
        self._modes_number = modes_number

    @property
    def particles_number(self) -> int:
        return self._particles_number

    @particles_number.setter
    def particles_number(self, particles_number) -> None:
        self._particles_number = particles_number

    @property
    def transmissivity(self) -> float:
        return self._transmissivity

    @transmissivity.setter
    def transmissivity(self, transmissivity: float) -> None:
        self._transmissivity = transmissivity

    def calculate_distribution(self) -> List[float]:
        """
        Computes the whole distribution.

        :return:
            The probabilities of all the output states.
        """
        return self.calculate_probabilities_of_outcomes(
            generate_possible_states(self._particles_number, self._modes_number, True)
        )

    def calculate_probabilities_of_outcomes(
        self, outcomes: List[Sequence[int]]
    ) -> List[float]:
        """
        Computes probabilities for the specified outputs.

        :param outcomes:
            The outcomes for which the probabilities should be computed.

        :return:
            The list of probabilities ordered in the same way as the outcomes.
        """
        all_outcomes: List[Tuple[int, ...]] = generate_possible_states(
            self._particles_number, self._modes_number, True
        )

        weights: List[float] = compute_binomial_weights(
            self._particles_number, self._transmissivity
        )

        probabilities: Dict[int, float] = {}

        for particles_number in range(self._particles_number + 1):
            probabilities[particles_number] = weights[particles_number] / sum(
                [1 for state in all_outcomes if sum(state) == particles_number]
            )

        return [probabilities[sum(state)] for state in outcomes]

    def get_outcomes_in_proper_order(self) -> List[Sequence[int]]:
        """
        Returns an ordered list of outcomes, where the ordering is the same as in
        calculate_distribution method.

        :return:
            Ordered list of outcomes where the order corresponds to the probabilities
            in the calculate distribution method.
        """
        return generate_possible_states(
            self._particles_number, self._modes_number, True
        )
