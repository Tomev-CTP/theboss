__author__ = "Tomasz Rybotycki"

"""
    This script contains an interface for the BS submatrices permanents calculators.
"""

import abc
from typing import List, Sequence


class BSSubmatricesPermanentCalculatorInterface(abc.ABC):
    """
    This is the interface class for the submatrices permanents calculator.
    """

    @abc.abstractmethod
    def compute_permanents(self) -> List[complex]:
        """
        Computes permanent of the previously specified matrix.

        .. warning::
            This is an abstract class, so this method is not implemented.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def matrix(self) -> Sequence[Sequence[complex]]:
        """
        A matrix describing the interferometer in the considered BS experiment.
        """
        raise NotImplementedError

    @matrix.setter
    @abc.abstractmethod
    def matrix(self, matrix: Sequence[Sequence[complex]]) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_state(self) -> Sequence[int]:
        """
        A Fock input state in the considered BS experiment.
        """
        raise NotImplementedError

    @input_state.setter
    @abc.abstractmethod
    def input_state(self, input_state: Sequence[int]) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_state(self) -> Sequence[int]:
        """
        A Fock output state in the considered BS experiment.
        """
        raise NotImplementedError

    @output_state.setter
    @abc.abstractmethod
    def output_state(self, output_state: Sequence[int]) -> None:
        raise NotImplementedError
