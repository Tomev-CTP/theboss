__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of submatrices calculation method as used
    in Clifford & Clifford version B of the algorithm. C&C algorithm uses Laplace
    expansion for the permanents in order to compute the set of probabilities in each
    step of the algorithm. Instead of computing each permanent separately, we can
    compute them all in one run, which is vastly more efficient.
    
    Instead of using Rysers formula, or Glynn formula, we base our approach on the
    Chin-Huh's formula, which takes into account possible bunching in the input states
    (or rather can be interpreted as that). 
"""

import abc
from numpy import complex128, ndarray, int64, array
from typing import List, Optional


class BSSubmatricesPermanentCalculatorInterface(abc.ABC):
    """
        This is the interface class for submatrices permanents calculator. For now
        BSCCCHSubmatricesPermanentCalculator is the only class of that kind, but it's
        possible that there will be more (based e.g. on Glynns formula that grants
        numerical stability.

        If the above happens, place the interface class in the separate file.
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


class BSSubmatricesPermanentCalculatorBase(BSSubmatricesPermanentCalculatorInterface,
                                           abc.ABC):
    """
        Base class for BSSubmatricesPermanentCalculator classes. It takes care of some
        boilerplate code.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        if output_state is None:
            output_state = array([], dtype=int64)
        if input_state is None:
            input_state = array([], dtype=int64)
        self._matrix = matrix
        self._input_state = input_state
        self._output_state = output_state

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: ndarray) -> None:
        self._matrix = matrix

    @property
    def input_state(self) -> ndarray:
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: ndarray) -> None:
        self._input_state = array(input_state, dtype=int64)

    @property
    def output_state(self) -> ndarray:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self._output_state = array(output_state, dtype=int64)