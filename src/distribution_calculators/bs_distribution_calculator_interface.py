__author__ = "Tomasz Rybotycki"

import abc
from dataclasses import dataclass
from typing import List, Iterable

from numpy import ndarray

from ..network_simulation_strategy import network_simulation_strategy


# TODO TR: MO doesn't approve of this class. It should be changed somehow.
@dataclass
class BosonSamplingExperimentConfiguration:
    interferometer_matrix: ndarray  # A matrix describing interferometer.
    initial_state: ndarray
    initial_number_of_particles: int
    number_of_modes: int
    number_of_particles_lost: int
    number_of_particles_left: int
    uniform_transmissivity: float = 0
    network_simulation_strategy: network_simulation_strategy = None
    # TODO TR:  Previously we've used the number of approximated modes instead of the
    #           the hierarchy level. There may be some errors after the changes, that
    #           we should fix.
    hierarchy_level: int = 0  # This is k from papers [1] and [2]


class BSDistributionCalculatorInterface(abc.ABC):
    """ Interface for boson sampling exact distribution calculators """

    @abc.abstractmethod
    def calculate_distribution(self) -> List[float]:
        """ One has to be able to calculate exact distribution with it """
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_probabilities_of_outcomes(self, outcomes: Iterable[Iterable[int]]) -> List[float]:
        """ This method allows one to compute probabilities of only selected outcomes. """
        raise NotImplementedError

    @abc.abstractmethod
    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        """ One also has to know the order of objects that returned probabilities correspond to """
        raise NotImplementedError
