__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to...
"""

from typing import Tuple, List, Sequence, Set, Dict
from theboss.boson_sampling_utilities.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)


class BS2SampleBootstrapKSValidator:
    """
    This class assumes
    """

    def __init__(
        self, input_state: Tuple[int, ...], matrix: Sequence[Sequence[complex]]
    ) -> None:
        self._input_state: Tuple[int, ...] = input_state
        self._matrix: Sequence[Sequence[complex]] = matrix

        raise NotImplementedError

    def validate(
        self,
        first_sample: Sequence[Tuple[int, ...]],
        second_sample: Sequence[Tuple[int, ...]],
    ) -> float:
        """

        :param first_sample:
        :param second_sample:
        :return:
        """
        distinct_samples: Set[Tuple[int, ...]] = set(first_sample)
        distinct_samples.update(second_sample)

        raise NotImplementedError

    def _compute_required_probabilities(
        self, distinct_samples: Set[Tuple[int, ...]]
    ) -> Dict[Tuple[int, ...], float]:
        raise NotImplementedError
