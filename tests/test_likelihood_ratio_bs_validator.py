__author__ = "Tomasz Rybotycki"

"""
    This script contains tests for the likelihood ratio BS validator. We use the same
    tests as were performed in [6], to ensure that they pass.
"""

import unittest
from numpy import isclose

from scipy.stats import unitary_group

from typing import Sequence, Tuple, List

from theboss.validation.likelihood_ratio_validator import LikelihoodRatioBSValidator

from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
    GeneralizedCliffordsBSimulationStrategy,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)

from theboss.distribution_calculators.bs_exact_distribution_with_uniform_losses import (
    BosonSamplingExperimentConfiguration,
    BSDistributionCalculatorWithUniformLosses,
)
from theboss.distribution_calculators.uniform_losses_uniform_distribution_calculator import (
    UniformDistributionCalculatorForUniformlyLossyBS,
)
from theboss.distribution_calculators.uniform_losses_distinguishable_particles_distribution_calculator import (
    UniformLossesDistinguishableParticlesDistributionCalculator,
)


class Test2SampleKSBSValidator(unittest.TestCase):
    """
    A test class for BS validators.
    """

    def setUp(self) -> None:
        self._modes_number: int = 4
        self._particles_number: int = self._modes_number

        self._matrix: Sequence[Sequence[complex]] = unitary_group.rvs(
            self._modes_number
        )

        input_state: List[int] = [0 for _ in range(self._modes_number)]
        input_state[0 : self._particles_number] = [
            1 for _ in range(self._particles_number)
        ]

        self._input_state: Tuple[int, ...] = tuple(input_state)

        permanent_calculator: RyserPermanentCalculator
        permanent_calculator = RyserPermanentCalculator(self._matrix, self._input_state)

        config: BosonSamplingExperimentConfiguration
        config = BosonSamplingExperimentConfiguration(
            self._matrix,
            self._input_state,
            0,
        )

        self._bs_distribution_calculator: BSDistributionCalculatorWithUniformLosses
        self._bs_distribution_calculator = BSDistributionCalculatorWithUniformLosses(
            config, permanent_calculator
        )

        self._distinguishable_distribution_calculator: UniformLossesDistinguishableParticlesDistributionCalculator
        self._distinguishable_distribution_calculator = (
            UniformLossesDistinguishableParticlesDistributionCalculator(
                config, permanent_calculator
            )
        )

        self._uniform_distribution_calculator: UniformDistributionCalculatorForUniformlyLossyBS
        self._uniform_distribution_calculator = (
            UniformDistributionCalculatorForUniformlyLossyBS(
                self._modes_number, self._particles_number, 1
            )
        )

        self._validator: LikelihoodRatioBSValidator
        self._validator = LikelihoodRatioBSValidator(
            self._bs_distribution_calculator,
            self._distinguishable_distribution_calculator,
        )

        self._samples_number: int = (
            500  # As in [6]. Large values causes numerical errors.
        )

        self._exact_sampler: GeneralizedCliffordsBSimulationStrategy
        self._exact_sampler = GeneralizedCliffordsBSimulationStrategy(
            permanent_calculator
        )

    def test_validator_for_exact_samples_against_uniform_hypothesis(self) -> None:
        """
        Test if KS validator recognizes that samples were drawn from the exact
        sampler, as the hypothesis suggests, or from the uniform distribution.
        """
        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        self._validator.alternative_hypothesis_probabilities_calculator = (
            self._uniform_distribution_calculator
        )

        hypothesis_probability: float = self._validator.validate(exact_samples)

        self.assertTrue(
            isclose(hypothesis_probability, 1),
            f"The validator wasn't as confident ({hypothesis_probability}) as expected (1)!",
        )

    def test_validator_for_exact_and_distinguishable_samples(self) -> None:
        """
        Test if KS validator recognizes that samples were drawn from the exact
        sampler, as the hypothesis suggests, or from the distinguishable particles'
        distribution.
        """
        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        hypothesis_probability: float = self._validator.validate(exact_samples)

        self.assertTrue(
            isclose(hypothesis_probability, 1),
            f"The validator wasn't as confident ({hypothesis_probability}) as expected (1)!",
        )

    def test_validator_for_same_hypotheses(self) -> None:
        """
        Test if validator return 0.5 for same hypotheses.

        TODO:   This was not tested in the [6], but seems reasonable to check. We should
                keep an eye on that.
        """
        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        self._validator.alternative_hypothesis_probabilities_calculator = (
            self._bs_distribution_calculator
        )

        self.assertTrue(
            isclose(self._validator.validate(exact_samples), 0.5),
            "Probability is not 0.5 for same hypotheses!",
        )

    def test_validator_for_different_reversed_hypothesis(self) -> None:
        """
        Test if KS validator recognizes that samples were drawn from the exact
        sampler, as the alternative hypothesis suggests, or from the distinguishable
        particles' distribution as the hypothesis suggests.
        """
        self._validator.hypothesis_probabilities_calculator = (
            self._distinguishable_distribution_calculator
        )
        self._validator.alternative_hypothesis_probabilities_calculator = (
            self._bs_distribution_calculator
        )

        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        hypothesis_probability: float = self._validator.validate(exact_samples)

        self.assertTrue(
            isclose(hypothesis_probability, 0),
            f"The validator was overconfident ({hypothesis_probability} vs. 0)!",
        )
