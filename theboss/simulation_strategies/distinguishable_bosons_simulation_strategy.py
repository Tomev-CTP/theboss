__author__ = "Tomasz Rybotycki"

"""
    This script contains a class for simulating BS experiments with fully
    distinguishable bosons.
"""


from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)

from theboss.boson_sampling_utilities import mode_assignment_to_mode_occupation

from typing import Sequence, List, Tuple

from numpy.random import random


class DistinguishableBosonsSimulationStrategy(SimulationStrategyInterface):
    """
    A simulation strategy for lossless BS simulation with fully distinguishable bosons.
    """

    def __init__(
        self, matrix: Sequence[Sequence[complex]],
    ):
        self._matrix: Sequence[Sequence[complex]] = matrix

    @property
    def matrix(self) -> Sequence[Sequence[complex]]:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: Sequence[Sequence[complex]]) -> None:
        self._matrix = matrix

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Simulates a BS experiment with fully distinguishable photons.

        :param input_state:
            A state at the input of the interferometer.
        :param samples_number:
            The number of samples to return.

        :return:
            Required number of samples of the BS experiment with the distinguishable
            bosons.
        """
        return [self._get_new_sample(input_state) for _ in range(samples_number)]

    def _get_new_sample(self, input_state: Sequence[int]) -> Tuple[int, ...]:
        """
        Generates a new sample for given input and set interferometer.

        :param input_state:
            A state at the input of the interferometer.

        :return:
            An output state from the BS experiment with distinguishable photons.
        """
        sample: Tuple[int, ...] = tuple()

        for mode in range(len(input_state)):

            if input_state[mode] == 0:
                continue

            probability_distribution: List[float] = self._get_probabilities(mode)

            for particle in range(input_state[mode]):
                sample += (self._sample_output_mode(probability_distribution),)

        return mode_assignment_to_mode_occupation(sample, len(input_state))

    def _get_probabilities(self, mode: int) -> List[float]:
        """
        Computes the probabilities of a particle, starting in the mode, to
        end up in any observed modes.

        :param mode:
            Mode in which the input particle(s) for the sampling is (are)>

        :return:
            The probabilities of the particle from the mode ending up in any
            of the output modes.
        """
        return [abs(self._matrix[mode][i]) ** 2 for i in range(len(self._matrix))]

    @staticmethod
    def _sample_output_mode(probabilities: List[float]) -> int:
        """
        Samples the mode

        :param probabilities:
            Probabilities of finding a particle in given mode.

        :return:
            The mode in which the particle was found.
        """
        r: float = random() * sum(probabilities)
        mode: int = 0

        probabilities_sum: float = probabilities[mode]

        while probabilities_sum < r:
            mode += 1
            probabilities_sum += probabilities[mode]

        return mode
