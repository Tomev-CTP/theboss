__author__ = "Tomasz Rybotycki"

"""
    This script contains tests for the 2 sample KS BS validator. We use the same tests
    as were performed in [6], to ensure that they pass.
"""

import unittest

from scipy.stats import unitary_group

from typing import Sequence, Tuple, List

from theboss.validation.bs_2_sample_bootstrap_ks_validator import (
    BS2SampleBootstrapKSValidator,
)

from theboss.simulation_strategies.uniform_sampler import UniformSamplingStrategy
from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
    GeneralizedCliffordsBSimulationStrategy,
)
from theboss.simulation_strategies.distinguishable_particles_simulation_strategy import (
    DistinguishableParticlesSimulationStrategy,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
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

        self._validator: BS2SampleBootstrapKSValidator
        self._validator = BS2SampleBootstrapKSValidator(self._input_state, self._matrix)

        self._samples_number: int = 20000  # As in [6].

        self._permanent_calculator: RyserPermanentCalculator
        self._permanent_calculator = RyserPermanentCalculator(
            self._matrix, self._input_state
        )

        self._exact_sampler: GeneralizedCliffordsBSimulationStrategy
        self._exact_sampler = GeneralizedCliffordsBSimulationStrategy(
            self._permanent_calculator
        )

        self._uniform_sampler: UniformSamplingStrategy
        self._uniform_sampler = UniformSamplingStrategy()

        self._distinguishable_sampler: DistinguishableParticlesSimulationStrategy
        self._distinguishable_sampler = DistinguishableParticlesSimulationStrategy(
            self._matrix
        )

    def test_validator_for_exact_and_uniform_sampler(self) -> None:
        """
        Test if KS validator distinguishes between uniform and exact sampler.
        """
        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        uniform_samples: List[Tuple[int, ...]] = self._uniform_sampler.simulate(
            self._input_state, self._samples_number
        )

        self.assertFalse(
            self._validator.validate(exact_samples, uniform_samples),
            "The KS test hypothesis was not rejected!",
        )

    def test_validator_for_exact_and_distinguishable_samples(self) -> None:
        """
        Test if KS validator distinguishes between distinguishable and exact sampler.
        """
        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        distinguishable_samples: List[
            Tuple[int, ...]
        ] = self._distinguishable_sampler.simulate(
            self._input_state, self._samples_number
        )

        self.assertFalse(
            self._validator.validate(exact_samples, distinguishable_samples),
            "The KS test hypothesis was not rejected!",
        )

    def test_validator_for_2_exact_samplers(self) -> None:
        """
        Test if KS validator distinguishes between exact and exact sampler.
        """
        exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        exact_samples_2: List[Tuple[int, ...]] = self._exact_sampler.simulate(
            self._input_state, self._samples_number
        )

        self.assertTrue(
            self._validator.validate(exact_samples, exact_samples_2),
            "The KS test hypothesis was rejected for the same samplers!",
        )
