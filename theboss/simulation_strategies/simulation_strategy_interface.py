__author__ = "Tomasz Rybotycki"

import abc
from typing import Sequence, List, Tuple


class SimulationStrategyInterface(abc.ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "simulate") and callable(subclass.simulate)

    @abc.abstractmethod
    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Simulate the lossy boson sampling experiment.

        :param input_state: Input state of the simulation.
        :param samples_number: Number of samples one wants to simulate.
        :return:
        """
        raise NotImplementedError
