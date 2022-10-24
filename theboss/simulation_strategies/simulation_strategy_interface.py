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

        .. warning::
            This is an abstract class. This method is not implemented.

        :param input_state:
            Input state of the simulation.
        :param samples_number:
            Number of samples one wants to simulate.

        :return:
            The samples from the BS experiment.
        """
        raise NotImplementedError
