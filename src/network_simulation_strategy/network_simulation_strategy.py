__author__ = "Tomasz Rybotycki"

import abc

from numpy import ndarray


class NetworkSimulationStrategy(abc.ABC):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, "simulate") and
                callable(subclass.simulate))

    @abc.abstractmethod
    def simulate(self, input_state: ndarray) -> ndarray:
        raise NotImplementedError
