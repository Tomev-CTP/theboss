__author__ = "Tomasz Rybotycki"

"""
    This script contains a class for simulating BS experiments with fully
    distinguishable bosons and uniform losses.
"""


from theboss.simulation_strategies.distinguishable_bosons_simulation_strategy import (
    DistinguishableBosonsSimulationStrategy,
)
from theboss.boson_sampling_utilities import apply_uniform_losses_to_the_state
from typing import Sequence, Tuple


class DistinguishableBosonsUniformLossesSimulationStrategy(
    DistinguishableBosonsSimulationStrategy
):
    """
    A simulation strategy for lossless BS simulation with fully distinguishable
    bosons and uniform losses.
    """

    def __init__(self, matrix: Sequence[Sequence[complex]], transmissivity: float = 1):
        super().__init__(matrix)
        self._transmissivity: float = transmissivity

    def _get_new_sample(self, input_state: Sequence[int]) -> Tuple[int, ...]:
        """
        Generates a new sample for given input and set interferometer.

        :param input_state:
            A state at the input of the interferometer.

        :return:
            An output state from the BS experiment with distinguishable photons.
        """
        lossy_input: Tuple[int, ...] = apply_uniform_losses_to_the_state(
            input_state, self._transmissivity
        )
        return super()._get_new_sample(lossy_input)
