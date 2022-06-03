__author__ = "Tomasz Rybotycki"

"""
    This file hold a factory for BS permanent calculators.
"""

import enum
from typing import Optional

from numpy import ndarray

from .bs_permanent_calculator_interface import BSPermanentCalculatorInterface
from .chin_huh_permanent_calculator import ChinHuhPermanentCalculator
from .classic_permanent_calculator import ClassicPermanentCalculator
from .parallel_chin_huh_permanent_calculator import ParallelChinHuhPermanentCalculator
from .ryser_permanent_calculator import RyserPermanentCalculator


class PermanentCalculatorType(enum.IntEnum):
    CLASSIC = enum.auto()
    CHIN_HUH = enum.auto()
    PARALLEL_CHIN_HUH = enum.auto()
    RYSER = enum.auto()


class BSPermanentCalculatorFactory:
    def __init__(
        self,
        matrix: Optional[ndarray],
        input_state: Optional[ndarray],
        output_state: Optional[ndarray],
        calculator_type: PermanentCalculatorType = PermanentCalculatorType.PARALLEL_CHIN_HUH,
    ):
        self._matrix = matrix
        self._input_state = input_state
        self._output_state = output_state

        self._calculator_type = calculator_type
        self._calculator_mapping = {
            PermanentCalculatorType.CLASSIC: self._generate_classic_permanent_calculator,
            PermanentCalculatorType.CHIN_HUH: self._generate_chin_huh_permanent_calculator,
            PermanentCalculatorType.PARALLEL_CHIN_HUH: self._generate_parallel_chin_huh_permanent_calculator,
            PermanentCalculatorType.RYSER: self._generate_ryser_permanent_calculator,
        }

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix: ndarray) -> None:
        self._matrix = new_matrix

    @property
    def input_state(self) -> ndarray:
        return self._input_state

    @input_state.setter
    def input_state(self, new_input_state: ndarray) -> None:
        self._input_state = new_input_state

    @property
    def output_state(self) -> ndarray:
        return self._output_state

    @output_state.setter
    def output_state(self, new_output_state: ndarray) -> None:
        self._output_state = new_output_state

    def generate_calculator(self) -> BSPermanentCalculatorInterface:
        """
        Generates the permanent as specified by the provided calculator type.
        """
        handler = self._calculator_mapping.get(
            self._calculator_type, self._generate_parallel_chin_huh_permanent_calculator
        )
        return handler()

    def _generate_classic_permanent_calculator(self) -> ClassicPermanentCalculator:
        return ClassicPermanentCalculator(
            matrix=self.matrix,
            input_state=self.input_state,
            output_state=self.output_state,
        )

    def _generate_chin_huh_permanent_calculator(self) -> ChinHuhPermanentCalculator:
        return ChinHuhPermanentCalculator(
            matrix=self.matrix,
            input_state=self.input_state,
            output_state=self.output_state,
        )

    def _generate_parallel_chin_huh_permanent_calculator(
        self,
    ) -> ParallelChinHuhPermanentCalculator:
        return ParallelChinHuhPermanentCalculator(
            matrix=self.matrix,
            input_state=self.input_state,
            output_state=self.output_state,
        )

    def _generate_ryser_permanent_calculator(self) -> RyserPermanentCalculator:
        return RyserPermanentCalculator(
            matrix=self.matrix,
            input_state=self.input_state,
            output_state=self.output_state,
        )
