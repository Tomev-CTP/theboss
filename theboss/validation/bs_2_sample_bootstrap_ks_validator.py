__author__ = "Tomasz Rybotycki"

"""
    This module contains the BS validator that uses 2-sample bootstrap KS test, as
    suggested in [6].
"""

from typing import Tuple, List, Sequence, Set, Dict
from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)
from numpy import log, sqrt

# TODO TR: Check the scipy package for 2-sample KS test.


class BS2SampleBootstrapKSValidator:
    """
    This class implements the 2-sample bootstrap KS test for BS validation in the same
    way that the authors did in [6].
    """

    def __init__(
        self,
        input_state: Tuple[int, ...],
        matrix: Sequence[Sequence[complex]],
        confidence_level: float = 0.1,
    ) -> None:
        self._input_state: Tuple[int, ...] = input_state
        self._matrix: Sequence[Sequence[complex]] = matrix
        self._confidence_level: float = self._trim_confidence_level(confidence_level)

    def _trim_confidence_level(self, confidence_level: float) -> float:
        """
        We expect the confidence level to be in ]0, 1[ interval. We'd like to trim
        it to this interval.

        :param confidence_level:
            User specified confidence level.

        :return:
            Confidence level trimmed to the proper value.
        """
        # Use 0.001 as the lowest value.
        if confidence_level < 0.001:
            return 0.001

        if confidence_level > 1:
            return 1

        return confidence_level

    @property
    def input_state(self) -> Tuple[int, ...]:
        """
        A Fock input state that was used in the experiments.
        """
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: Tuple[int, ...]) -> None:
        self._input_state = input_state

    @property
    def matrix(self) -> Sequence[Sequence[complex]]:
        """
        The interferometer matrix that was used in the experiment.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: Sequence[Sequence[complex]]) -> None:
        self._matrix = matrix

    @property
    def confidence_level(self) -> float:
        """
        The desired confidence level of the validator.
        """
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, confidence_level: float) -> None:
        self._confidence_level = self._trim_confidence_level(confidence_level)

    def validate(
        self,
        first_sample: Sequence[Tuple[int, ...]],
        second_sample: Sequence[Tuple[int, ...]],
    ) -> bool:
        """
        Checks if given samples were drawn from the same BS distribution.

        .. note::
            The samples can have arbitrary (different) sizes.

        :param first_sample:
            One of the samples to compare.
        :param second_sample:
            One of the samples to compare.

        :return:
            ``True`` if the samples were sampled from the same distribution (with
            specified confidence). Else ``False``.
        """

        distinct_samples: Set[Tuple[int, ...]] = set(first_sample)
        distinct_samples.update(second_sample)

        permanent_coefficients = self._compute_required_permanent_coefficients(
            distinct_samples
        )

        first_sample_coefficients: List[float] = [
            permanent_coefficients[s] for s in first_sample
        ]
        second_sample_coefficients: List[float] = [
            permanent_coefficients[s] for s in second_sample
        ]

        ks_statistic: float = self._compute_ks_statistic(
            first_sample_coefficients, second_sample_coefficients
        )

        value_for_comparison: float = self._compute_c_constant()

        n: int = len(first_sample)
        m: int = len(second_sample)

        value_for_comparison *= sqrt((n + m) / (n * m))

        return ks_statistic <= value_for_comparison

    def _compute_required_permanent_coefficients(
        self, distinct_samples: Set[Tuple[int, ...]]
    ) -> Dict[Tuple[int, ...], float]:
        """
        Computes :math:`|Perm(A_s)|^2` for each distinct sample that was given in
        the input of the validator.

        :return:
            A dict with samples and the values of :math:`|Perm(A_s)|^2` corresponding
            to them.
        """
        permanent_calculator: RyserPermanentCalculator = RyserPermanentCalculator(
            matrix=self._matrix, input_state=self._input_state, output_state=None
        )

        permanent_coefficients: Dict[Tuple[int, ...], float] = {}

        for sample in distinct_samples:
            permanent_calculator.output_state = sample
            permanent_coefficients[sample] = -log(
                abs(permanent_calculator.compute_permanent()) ** 2
            )

        return permanent_coefficients

    def _compute_ks_statistic(
        self,
        first_sample_coefficients: List[float],
        second_sample_coefficients: List[float],
    ) -> float:
        """
        Computes the value of 2-sample KS statistics, given by

        ..math::
            \\sup_x |F_1(x) - F_2(x)|,

        where :math:`F_s` is the empirical distribution function for sample :math:`s`.

        :param first_sample_coefficients:
            The permanent coefficients corresponding to the first sample.
        :param second_sample_coefficients:
            The permanent coefficients corresponding to the second sample.

        :return:
            The value of 2-sample KS statistics.
        """
        ks_statistic: float = 0

        distinct_coefficients: Set[float] = set(first_sample_coefficients)
        distinct_coefficients.update(second_sample_coefficients)

        for coefficient in distinct_coefficients:
            ks_statistic = max(
                ks_statistic,
                abs(
                    self._compute_empirical_distribution_function_value_in_x(
                        first_sample_coefficients, coefficient
                    )
                    - self._compute_empirical_distribution_function_value_in_x(
                        second_sample_coefficients, coefficient
                    )
                ),
            )

        return ks_statistic

    def _compute_empirical_distribution_function_value_in_x(
        self, sample: List[float], x: float
    ) -> float:
        """
        Computes the value of empirical distribution function in point :math:`x`.
        The empirical distribution function is given by
        .. math::
            F(x) = \\frac{#(elements in the sample \\leq x)}{#sample}.

        :param sample:
            The sample for which empirical distribution function will be computed.
        :param x:
            The value with which the elements in the samples will be compared.
            Essentially this is the point in which the value of empirical distribution
            function will be evaluated.

        :return:
            The value of empirical distribution function in :math:`x`.
        """
        return sum(1 for element in sample if element <= x) / len(sample)

    def _compute_c_constant(self) -> float:
        """
        Computes the :math`c` constant part of the tests' inequality.

        :return:
            The :math:`c` constant part of the tests' inequality.
        """
        return sqrt(-0.5 * log(self._confidence_level / 2))
