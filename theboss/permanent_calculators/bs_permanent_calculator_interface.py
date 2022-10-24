__author__ = "Tomasz Rybotycki"

"""
    This file holds an interface for Boson Sampling (BS) Permanent Calculators.
    BS permanent calculation is different from the standard permanent calculation in a
    way, that it requires computing a reduced matrix (called in BS the effective
    scattering matrix) that depends on the input and the output state.
     
    In order to compute the permanent of a given matrix one should set input and output
    states to [1, ... , 1]. 
"""

import abc

from typing import Sequence


class BSPermanentCalculatorInterface(abc.ABC):
    @abc.abstractmethod
    def compute_permanent(self) -> complex:
        """
        Computes permanent of the previously specified matrix.

        .. warning::
            This is an abstract class. This method is not implemented.

        :return:
            The permanent of the specified (possibly reduced) matrix.
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
