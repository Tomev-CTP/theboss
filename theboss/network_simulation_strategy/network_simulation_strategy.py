__author__ = "Tomasz Rybotycki"

import abc

from typing import Sequence


class NetworkSimulationStrategy(abc.ABC):
    """
    An interface class for network simulation strategies. They are used for the
    full mean-field method simulations, as presented in [1].
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "simulate") and callable(subclass.simulate)

    @abc.abstractmethod
    def simulate(self, input_state: Sequence[int]) -> Sequence[Sequence[complex]]:
        """
        Main method of all NetworkSimulation classes.

        .. warning::
            This is an abstract class, so the method is not implemented.

        :param input_state:
            An input Fock state of the simulations.

        :return:
            The evolved (approximated) input state.
        """
        raise NotImplementedError
