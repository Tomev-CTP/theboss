__author__ = "Tomasz Rybotycki"

"""
    The aim of this script is to provide some basic tests for the uniform distribution
    calculators from the BS outputs distribution.
"""

import unittest
from numpy import isclose

from theboss.distribution_calculators.uniform_losses_uniform_distribution_calculator import (
    UniformDistributionCalculatorForUniformlyLossyBS,
)

from theboss.distribution_calculators.uniform_distribution_calculator_for_lossy_bs import (
    UniformDistributionCalculatorForLossyBS,
)


class TestUniformDistributionCalculators(unittest.TestCase):

    """
    Test basic tests for the uniform distribution calculators. Basically this only
    checks if the distributions sums up to 1.
    """

    def setUp(self) -> None:
        self._modes_number: int = 4
        self._particles_number: int = 4
        self._uniform_transmission_probability: float = 0.3

    def test_uniform_losses_calc_for_no_losses(self) -> None:
        """
        Test uniform distribution calculator for no losses.
        """
        calc = UniformDistributionCalculatorForUniformlyLossyBS(
            self._modes_number, self._particles_number, 1
        )
        distribution = calc.calculate_distribution()

        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_uniform_losses_calc_for_uniform_losses(self) -> None:
        """
        Test uniformly lossy uniform distribution calculator.
        """
        calc = UniformDistributionCalculatorForUniformlyLossyBS(
            self._modes_number,
            self._particles_number,
            self._uniform_transmission_probability,
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )

    def test_general_losses_calc(self) -> None:
        """
        Test generally lossy uniform distribution calculator.
        """
        calc = UniformDistributionCalculatorForLossyBS(
            self._modes_number, self._particles_number
        )
        distribution = calc.calculate_distribution()
        self.assertTrue(
            isclose(sum(distribution), 1), f"Distribution sum is {sum(distribution)}"
        )
