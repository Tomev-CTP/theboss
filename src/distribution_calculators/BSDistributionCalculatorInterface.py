__author__ = "Tomasz Rybotycki"

import abc
from dataclasses import dataclass
from typing import List

from numpy import ndarray

from src.network_simulation_strategy import NetworkSimulationStrategy


# TODO TR: This class should be placed in separate folder
@dataclass
class BosonSamplingExperimentConfiguration:
    interferometer_matrix: ndarray  # A matrix describing interferometer.
    initial_state: ndarray
    initial_number_of_particles: int
    number_of_modes: int
    number_of_particles_lost: int
    number_of_particles_left: int
    uniform_transmissivity: float = 0
    network_simulation_strategy: NetworkSimulationStrategy = None
    lossy_modes_number: int = 0


class BSDistributionCalculatorInterface(abc.ABC):
    """ Interface for boson sampling exact distribution calculators """

    @abc.abstractmethod
    def calculate_distribution(self) -> List[float]:
        """ One has to be able to calculate exact distribution with it """
        raise NotImplementedError

    @abc.abstractmethod
    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        """ One also has to know the order of objects that returned probabilities correspond to """
        raise NotImplementedError
