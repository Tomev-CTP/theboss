__author__ = "Tomasz Rybotycki"

"""
    This script contains an implementation of the likelihood ratio test for BS
    validation. We will use similar method as in [6], but instead of computing
    permanents and performing some normalization, we will just use plain probabilities
    computed using specified BS distribution calculators.
"""

from typing import Sequence, Tuple, List, Dict

from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BSDistributionCalculatorInterface,
)


class LikelihoodRatioBSValidator:
    """
    A BS validator using the likelihood ratio test, similar to the one used in [6].
    """

    def __init__(
        self,
        hypothesis_probabilities_calculator: BSDistributionCalculatorInterface,
        alternative_hypothesis_probabilities_calculator: BSDistributionCalculatorInterface,
    ):
        self._hypothesis_probabilities_calculator: BSDistributionCalculatorInterface
        self._hypothesis_probabilities_calculator = hypothesis_probabilities_calculator

        self._alternative_hypothesis_probabilities_calculator: BSDistributionCalculatorInterface
        self._alternative_hypothesis_probabilities_calculator = (
            alternative_hypothesis_probabilities_calculator
        )

    @property
    def hypothesis_probabilities_calculator(self) -> BSDistributionCalculatorInterface:
        """
        Probabilities calculator for the hypothesis.
        """
        return self._hypothesis_probabilities_calculator

    @hypothesis_probabilities_calculator.setter
    def hypothesis_probabilities_calculator(
        self, hypothesis_probabilities_calculator: BSDistributionCalculatorInterface
    ) -> None:
        self._hypothesis_probabilities_calculator = hypothesis_probabilities_calculator

    @property
    def alternative_hypothesis_probabilities_calculator(
        self,
    ) -> BSDistributionCalculatorInterface:
        """
        Probabilities calculator for the alternative hypothesis.
        """
        return self._alternative_hypothesis_probabilities_calculator

    @alternative_hypothesis_probabilities_calculator.setter
    def alternative_hypothesis_probabilities_calculator(
        self,
        alternative_hypothesis_probabilities_calculator: BSDistributionCalculatorInterface,
    ) -> None:
        self._alternative_hypothesis_probabilities_calculator = (
            alternative_hypothesis_probabilities_calculator
        )

    def validate(self, samples: Sequence[Tuple[int, ...]]) -> float:
        """
        Main method of the validator. It returns the probability that the samples
        were drawn according to the hypothesis, not the alternative. The returned
        probability is based on the likelihood ratio.

        .. warning::
            It may generate overflow problems for some (large) ``chi`` values. In such
            cases the ``samples`` size should be reduced.

        :param samples:
            Samples to be validated. We check if they were drawn according to the
            hypothesis.

        :return:
            Probability that the samples were drawn from the hypothesis.
        """
        distinct_samples: List[Tuple[int, ...]] = list(set(samples))

        hypothesis_probabilities: Dict[Tuple[int, ...], float] = {}
        alternative_hypothesis_probabilities: Dict[Tuple[int, ...], float] = {}

        probs: List[
            float
        ] = self._hypothesis_probabilities_calculator.calculate_probabilities_of_outcomes(
            distinct_samples
        )

        a_probs: List[
            float
        ] = self._alternative_hypothesis_probabilities_calculator.calculate_probabilities_of_outcomes(
            distinct_samples
        )

        for i in range(len(distinct_samples)):
            hypothesis_probabilities[distinct_samples[i]] = probs[i]
            alternative_hypothesis_probabilities[distinct_samples[i]] = a_probs[i]

        chi: float = 1  # A normalization factor.

        for sample in samples:
            # TODO TR: This may produce overflow problems for large values.
            chi *= (
                hypothesis_probabilities[sample]
                / alternative_hypothesis_probabilities[sample]
            )

        return chi / (1 + chi)
