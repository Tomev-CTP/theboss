__author__ = "Tomasz Rybotycki"

"""
    This script contains the tests for cluster-based validator.
"""

import unittest

from scipy.stats import unitary_group

from typing import Sequence, Tuple, List

from theboss.validation.cluster_based_bs_validator import ClusterBasedBSValidator

from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
    GeneralizedCliffordsBSimulationStrategy,
)
from theboss.simulation_strategies.distinguishable_particles_simulation_strategy import (
    DistinguishableParticlesSimulationStrategy,
)

from theboss.permanent_calculators.ryser_permanent_calculator import (
    RyserPermanentCalculator,
)

from tqdm import tqdm


class TestClusterBasedBSValidator(unittest.TestCase):
    """
    A test class for cluster-based BS validators.
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

        self._validator: ClusterBasedBSValidator
        self._validator = ClusterBasedBSValidator(self._modes_number)

        self._experiments_number: int = 100  # As in [7].
        self._samples_number: int = 500  # As in [7].

        self._permanent_calculator: RyserPermanentCalculator
        self._permanent_calculator = RyserPermanentCalculator(
            self._matrix, self._input_state
        )

        self._exact_sampler: GeneralizedCliffordsBSimulationStrategy
        self._exact_sampler = GeneralizedCliffordsBSimulationStrategy(
            self._permanent_calculator
        )

        self._distinguishable_sampler: DistinguishableParticlesSimulationStrategy
        self._distinguishable_sampler = DistinguishableParticlesSimulationStrategy(
            self._matrix
        )

    def test_validator_for_exact_and_distinguishable_samples(self) -> None:
        """
        Test if KS validator distinguishes between distinguishable and exact sampler.
        """
        successes_number: int = 0

        for _ in range(self._experiments_number):

            exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
                self._input_state, self._samples_number
            )

            distinguishable_samples: List[
                Tuple[int, ...]
            ] = self._distinguishable_sampler.simulate(
                self._input_state, self._samples_number
            )

            if self._validator.validate_majority_voting(
                exact_samples, distinguishable_samples
            ):
                successes_number += 1

        self.assertFalse(
            successes_number > self._experiments_number // 2,
            "The cluster-based validator couldn't discern the sample from the the real distribution and the distinguishable-particles one!",
        )

    def test_validator_for_2_exact_samplers(self) -> None:
        """
        Test if cluster-based validator [7] distinguishes between exact and exact
        sampler.
        """
        successes_number: int = 0

        for _ in tqdm(range(self._experiments_number)):

            exact_samples: List[Tuple[int, ...]] = self._exact_sampler.simulate(
                self._input_state, self._samples_number
            )

            exact_samples_2: List[Tuple[int, ...]] = self._exact_sampler.simulate(
                self._input_state, self._samples_number
            )

            if self._validator.validate_majority_voting(exact_samples, exact_samples_2):
                successes_number += 1

        self.assertTrue(
            successes_number > self._experiments_number // 2,
            f"The cluster-based validator accepted {successes_number} out of {self._experiments_number} experiments!",
        )
