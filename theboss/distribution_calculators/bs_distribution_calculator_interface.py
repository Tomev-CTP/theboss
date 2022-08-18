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

    interferometer_matrix: Sequence[
        Sequence[complex]
    ]  # A matrix describing interferometer.
    initial_state: Sequence[int]
    initial_number_of_particles: int
    number_of_modes: int
    number_of_particles_lost: int
    number_of_particles_left: int
    uniform_transmissivity: float = 1
    network_simulation_strategy: NetworkSimulationStrategy = None
    hierarchy_level: int = 0  # This is k from paper[2]

    def __init__(
        self,
        interferometer_matrix: Sequence[Sequence[complex]],
        initial_state: Sequence[int],
        number_of_particles_lost: int,
        uniform_transmissivity: float = 1,
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
        self.uniform_transmissivity = uniform_transmissivity
        self.network_simulation_strategy = network_simulation_strategy
        self.hierarchy_level = hierarchy_level


class BSDistributionCalculatorInterface(abc.ABC):
    """Interface for boson sampling exact distribution calculators"""

    @abc.abstractmethod
    def calculate_distribution(self) -> List[float]:
        """One has to be able to calculate exact distribution with it"""
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_probabilities_of_outcomes(
        self, outcomes: List[Sequence[int]]
    ) -> List[float]:
        """
        This method allows one to compute probabilities of only selected outcomes.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_outcomes_in_proper_order(self) -> List[Tuple[int, ...]]:
        """
        One also has to know the order of objects that returned probabilities correspond
        to.
        """
        raise NotImplementedError
