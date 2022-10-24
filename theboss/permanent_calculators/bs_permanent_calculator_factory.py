__author__ = "Tomasz Rybotycki"

"""
    This file hold a factory for BS permanent calculators.
"""

import enum
from typing import Optional, Sequence, Dict, Callable

from theboss.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)
from theboss.permanent_calculators.classic_permanent_calculator import (
    ClassicPermanentCalculator,
)
from theboss.permanent_calculators.glynn_gray_permanent_calculator import (
    GlynnGrayPermanentCalculator,
)
from theboss.permanent_calculators.chin_huh_permanent_calculator import (
    ChinHuhPermanentCalculator,
)
from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)


class PermanentCalculatorType(enum.IntEnum):
    """
    An enumerator of the currently available permanent calculators.

    1. Classic permanent calculator computes the desired sub-matrix and the permanent of it.

    2. Glynn permanent calculator uses Glynn's formula.

    3. Chin-Huh permanent calculator uses Chin & Huh's formula.

    4. Ryser permanent calculator uses Ryser's formula.
    """

    CLASSIC = enum.auto()
    GLYNN = enum.auto()
    CHIN_HUH = enum.auto()
    RYSER = enum.auto()


class BSPermanentCalculatorFactory:
    """
    A permanent calculators factory. The idea behind this class was to create a class
    that one would include in more complex experiments, to effortlessly change the
    permanent calculators on the fly.

    .. note::
        In practice it turned out not to be used that much.
    """

    def __init__(
        self,
        matrix: Optional[Sequence[Sequence[complex]]],
        input_state: Optional[Sequence[int]],
        output_state: Optional[Sequence[int]],
        calculator_type: PermanentCalculatorType = PermanentCalculatorType.RYSER,
    ):
        self._matrix: Sequence[Sequence[complex]] = matrix
        self._input_state: Sequence[int] = input_state
        self._output_state: Sequence[int] = output_state

        self._calculator_type: PermanentCalculatorType = calculator_type
        self._calculator_mapping: Dict[
            PermanentCalculatorType, Callable[[], BSPermanentCalculatorInterface]
        ] = {
            PermanentCalculatorType.CLASSIC: self._generate_classic_permanent_calculator,
            PermanentCalculatorType.GLYNN: self._generate_glynn_permanent_calculator,
            PermanentCalculatorType.CHIN_HUH: self._generate_chin_huh_permanent_calculator,
            PermanentCalculatorType.RYSER: self._generate_ryser_permanent_calculator,
        }

    @property
    def matrix(self) -> Sequence[Sequence[complex]]:
        """
        A matrix representing the interferometer matrix in the considered BS experiment.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix: Sequence[Sequence[complex]]) -> None:
        self._matrix = new_matrix

    @property
    def input_state(self) -> Sequence[int]:
        """
        The Fock input state in the considered BS experiment.
        """
        return self._input_state

    @input_state.setter
    def input_state(self, new_input_state: Sequence[int]) -> None:
        self._input_state = new_input_state

    @property
    def output_state(self) -> Sequence[int]:
        """
        The Fock output state of the considered BS experiment.
        """
        return self._output_state

    @output_state.setter
    def output_state(self, new_output_state: Sequence[int]) -> None:
        self._output_state = new_output_state

    @property
    def calculator_type(self) -> PermanentCalculatorType:
        """
        Permanent calculator type currently returned by the factory.
        """
        return self._calculator_type

    @calculator_type.setter
    def calculator_type(self, new_calculator_type: PermanentCalculatorType) -> None:
        self._calculator_type = new_calculator_type

    def generate_calculator(self) -> BSPermanentCalculatorInterface:
        """
        Generates the permanent calculator as specified by the ``calculator_type``
        provided in the factory constructor.
        """
        handler = self._calculator_mapping.get(
            self._calculator_type, self._generate_chin_huh_permanent_calculator
        )
        return handler()

    def _generate_classic_permanent_calculator(self) -> ClassicPermanentCalculator:
        return ClassicPermanentCalculator(
            matrix=self.matrix,
            input_state=self.input_state,
            output_state=self.output_state,
        )

    def _generate_glynn_permanent_calculator(self) -> GlynnGrayPermanentCalculator:
        return GlynnGrayPermanentCalculator(
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

    def _generate_ryser_permanent_calculator(self) -> RyserPermanentCalculator:
        return RyserPermanentCalculator(
            matrix=self.matrix,
            input_state=self.input_state,
            output_state=self.output_state,
        )
