__author__ = "Tomasz Rybotycki"

"""
    This script contains an interface for the BS submatrices permanents calculators.
"""

import abc
from typing import List
from numpy import ndarray, complex128


class BSSubmatricesPermanentCalculatorInterface(abc.ABC):
    """
    This is the interface class for submatrices permanents calculator.
    """

    @abc.abstractmethod
    def compute_permanents(self) -> List[complex128]:
        """Computes permanent of a matrix given before."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def matrix(self) -> ndarray:
        raise NotImplementedError

    @matrix.setter
    @abc.abstractmethod
    def matrix(self, matrix: ndarray) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def input_state(self) -> ndarray:
        raise NotImplementedError

    @input_state.setter
    @abc.abstractmethod
    def input_state(self, input_state: ndarray) -> None:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def output_state(self) -> ndarray:
        raise NotImplementedError

    @output_state.setter
    @abc.abstractmethod
    def output_state(self, output_state: ndarray) -> None:
        raise NotImplementedError
