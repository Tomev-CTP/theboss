__author__ = "Tomasz Rybotycki"

from typing import List, Sequence

"""
    This script contains the implementation of the probabilities calculator for the 
    uniform sampler from the non-uniformly lossy BS distribution. This is very naive
    approach as it assumes equal probability for each lossy state (so the average
    number of particles will most likely be off). 
"""

from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BSDistributionCalculatorInterface,
)
from theboss.boson_sampling_utilities import (
    bosonic_space_dimension,
    generate_possible_states,
)


class UniformDistributionCalculatorForLossyBS(BSDistributionCalculatorInterface):
    """
    A class for computing the output probabilities of the non-uniformly lossy uniform
    sampling from BS distribution with a fixed number of initial particles and modes.
    """

    def __init__(self, modes_number: int, particles_number: int) -> None:
        self._modes_number: int = modes_number
        self._particles_number: int = particles_number

    @property
    def modes_number(self) -> int:
        """
        The number of modes in the considered BS experiment.
        """
        return self._modes_number

    @modes_number.setter
    def modes_number(self, modes_number) -> None:
        self._modes_number = modes_number

    @property
    def particles_number(self) -> int:
        """
        The number of initial particles in the considered BS experiment.
        """
        return self._particles_number

    @particles_number.setter
    def particles_number(self, particles_number) -> None:
        self._particles_number = particles_number

    def calculate_distribution(self) -> List[float]:
        """
        Computes the whole distribution.

        .. note::
            In the case of uniform sampling it's a flat distribution over all possible
            BS states specified by the experiment configuration.

        :return:
            The probabilities of all the possible lossy output states.
        """
        return self.calculate_probabilities_of_outcomes(
            generate_possible_states(self._particles_number, self._modes_number, True)
        )

    def calculate_probabilities_of_outcomes(
        self, outcomes: List[Sequence[int]]
    ) -> List[float]:
        """
        Computes probabilities for the specified outputs.

        .. note::
            In the uniform sampling case, probabilities of all outputs are the same.

        :param outcomes:
            The outcomes for which the probabilities should be computed.

        :return:
            The list of probabilities ordered in the same way as the outcomes.
        """
        outcomes_number: int = bosonic_space_dimension(
            self._particles_number, self._modes_number, True
        )

        return [1 / outcomes_number] * len(outcomes)

    def get_outcomes_in_proper_order(self) -> List[Sequence[int]]:
        """
        Returns an ordered list of outcomes, where the ordering is the same as in
        calculate_distribution method.

        :return:
            Ordered list of outcomes where the order corresponds to the probabilities
            in the ``calculate_distribution`` method.
        """
        return generate_possible_states(
            self._particles_number, self._modes_number, True
        )
