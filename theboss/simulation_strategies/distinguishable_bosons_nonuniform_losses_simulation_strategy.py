__author__ = "Tomasz Rybotycki"

"""
    This script contains a class for simulating BS experiments with fully
    distinguishable bosons and nonuniform losses.
"""


from theboss.simulation_strategies.distinguishable_bosons_simulation_strategy import (
    DistinguishableBosonsSimulationStrategy,
)
from typing import Sequence, Tuple, List
from theboss.boson_sampling_utilities.boson_sampling_utilities import (
    prepare_interferometer_matrix_in_expanded_space,
)


class DistinguishableBosonsNonuniformLossesSimulationStrategy(
    DistinguishableBosonsSimulationStrategy
):
    """
    A simulation strategy for BS simulation with fully distinguishable
    bosons and non-uniform losses.
    """

    def __init__(self, matrix: Sequence[Sequence[complex]]):
        super().__init__(prepare_interferometer_matrix_in_expanded_space(matrix))
        self._initial_matrix: Sequence[Sequence[complex]] = matrix

    @property
    def matrix(self) -> Sequence[Sequence[complex]]:
        return self._initial_matrix

    @matrix.setter
    def matrix(self, matrix: Sequence[Sequence[complex]]) -> None:
        self._matrix = prepare_interferometer_matrix_in_expanded_space(matrix)
        self._initial_matrix = matrix

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Simulates a BS experiment with fully distinguishable photons and
        non-uniform losses.

        :param input_state:
            A state at the input of the interferometer.
        :param samples_number:
            The number of samples to return.

        :return:
            Required number of samples of the BS experiment with the distinguishable
        """
        expanded_input: List[int] = list(input_state)
        expanded_input.extend([0 for _ in input_state])
        expanded_samples: List[Tuple[int, ...]] = [
            self._get_new_sample(expanded_input) for _ in range(samples_number)
        ]
        return [sample[0 : len(input_state)] for sample in expanded_samples]
