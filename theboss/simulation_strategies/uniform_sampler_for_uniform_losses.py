__author__ = "Tomasz Rybotycki"

"""
This script contains an implementation of a class for uniformly lossy uniform sampling
of the BS output states basing on the input states. It has been implemented for the
tests and validators.
"""

from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)

from typing import List, Tuple, Sequence, Dict
from theboss.boson_sampling_utilities import (
    generate_possible_states,
    compute_binomial_weights,
)
from numpy.random import randint, random


class UniformSamplingWithUniformLossesStrategy(SimulationStrategyInterface):
    """
        A class for uniformly lossy uniform sampling from the proper BS output states.
    """

    def __init__(self, transmission_probability: float = 1) -> None:
        self._transmission_probability: float = transmission_probability

    @property
    def transmission_probability(self) -> float:
        return self._transmission_probability

    @transmission_probability.setter
    def transmission_probability(self, transmission_probability: float) -> None:
        self._transmission_probability = transmission_probability

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Draws required number of elements from the BS output space.

        :param input_state:
            A state that would be at the input of the interferometer. It serves as a
            source of knowledge about particles number and the modes number, thus
            generally about the output states space.
        :param samples_number:
            The number of samples to return.

        :return:
            Uniformly sampled required number of output states.
        """
        binomial_weights: List[float] = compute_binomial_weights(
            sum(input_state), self._transmission_probability
        )

        possible_output_states: Dict[int, List[Tuple[int, ...]]] = {}

        for n in range(sum(input_state) + 1):
            possible_output_states[n] = generate_possible_states(n, len(input_state))

        samples: List[Tuple[int, ...]] = []

        for _ in range(samples_number):
            output_particles_number: int = self._sample_output_particles_number(
                binomial_weights
            )

            current_lossy_states: List[Tuple[int, ...]] = possible_output_states[
                output_particles_number
            ]

            samples.append(current_lossy_states[randint(0, len(current_lossy_states))])

        return samples

    @staticmethod
    def _sample_output_particles_number(weights: List[float]) -> int:
        """
        Checks how many particles should be found in the input in the next sample.

        :param weights:
            Weights according to which sampling will be performed.

        :return:
            The number of particles in the next sample.
        """
        r: float = random()
        output_particles_number: int = 0
        weights_sum: float = weights[output_particles_number]

        while weights_sum < r:
            output_particles_number += 1
            weights_sum += weights[output_particles_number]

        return output_particles_number
