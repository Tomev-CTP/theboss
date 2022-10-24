__author__ = "Tomasz Rybotycki"

import abc
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from theboss.network_simulation_strategy.network_simulation_strategy import (
    NetworkSimulationStrategy,
)


# TODO TR: MO doesn't approve of this class. It should be changed somehow.
@dataclass
class BosonSamplingExperimentConfiguration:
    """
    A dataclass for storing the simulated experiment information in one place.
    """

    def __init__(
        self,
        interferometer_matrix: Sequence[Sequence[complex]],
        initial_state: Sequence[int],
        number_of_particles_lost: int,
        uniform_transmission_probability: float = 1,
        network_simulation_strategy: NetworkSimulationStrategy = None,
        hierarchy_level: int = 0,
    ) -> None:
        self.interferometer_matrix = interferometer_matrix
        self.initial_state = initial_state
        self.initial_number_of_particles = sum(initial_state)
        self.number_of_modes = len(initial_state)
        self.number_of_particles_lost = number_of_particles_lost
        self.number_of_particles_left = (
            self.initial_number_of_particles - number_of_particles_lost
        )
        self.uniform_transmission_probability = uniform_transmission_probability
        self.network_simulation_strategy = network_simulation_strategy
        self.hierarchy_level = hierarchy_level


class BSDistributionCalculatorInterface(abc.ABC):
    """Interface for boson sampling exact distribution calculators"""

    @abc.abstractmethod
    def calculate_distribution(self) -> List[float]:
        """
        A method for computing the whole distribution.

        .. warning::
            It's an abstract class. This method is not implemented.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_probabilities_of_outcomes(
        self, outcomes: List[Sequence[int]]
    ) -> List[float]:
        """
        A method for computing the probability of the specified outcomes.

        .. warning::
            It's an abstract class. This method is not implemented.

        :param outcomes:
            The outcomes for which the probabilities should be computed.

        :return:
            The probabilities of the ``outcomes`` in the same order.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_outcomes_in_proper_order(self) -> List[Tuple[int, ...]]:
        """
        Since the ``calculate_distribution`` method has no specified ordering of the
        results, this method should be able to return that ordering.

        .. warning::
            It's an abstract class. This method is not implemented.

        :return:
            The order of the outcomes in corresponding to the probabilities returned
            by the ``calculate_distribution`` method.
        """
        raise NotImplementedError
