__author__ = "Tomasz Rybotycki"

"""
    This file holds an interface for Boson Sampling Permanent Calculators. Boson Sampling (BS) Permanent calculation is
    different in that way, that it requires computing (or rather BS is described by) effective scattering matrix thus
    the permanent returned is not necessarily the permanent of a specified matrix. In order to compute the permanent
    of a given matrix one should specify input and output states as [1, ... , 1]. 
"""

import abc

from numpy import complex128, ndarray


class BSPermanentCalculatorInterface(abc.ABC):

    @abc.abstractmethod
    def compute_permanent(self) -> complex128:
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
