__author__ = "Tomasz Rybotycki"

from typing import List, Sequence, Tuple

from numpy import arange, ndarray
from numpy.random import choice
from scipy.special import binom

from theboss.boson_sampling_simulator import BosonSamplingSimulator
from theboss.simulation_strategies.fixed_loss_simulation_strategy import (
    FixedLossSimulationStrategy,
)
from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)


class UniformLossSimulationStrategy(SimulationStrategyInterface):
    """
    An implementation of a strategy for simulating BS experiments with uniform losses.
    """

    def __init__(
        self,
        interferometer_matrix: Sequence[Sequence[complex]],
        number_of_modes: int,
        transmissivity: float,
    ) -> None:
        self.interferometer_matrix: Sequence[Sequence[complex]] = interferometer_matrix
        self.number_of_modes: int = number_of_modes
        self.transmissivity: float = transmissivity

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """


        :param input_state:

        :param samples_number:

        :return:
        """
        initial_number_of_particles = int(sum(input_state))

        # Using n, eta, l notation from the paper.
        n: int = initial_number_of_particles
        eta: float = self.transmissivity

        separable_states_weights: List[float] = [
            pow(eta, l) * binom(n, l) * pow(1.0 - eta, n - l) for l in range(n + 1)
        ]

        samples: List[Tuple[int, ...]] = []

        while len(samples) < samples_number:
            number_of_particles_left_in_selected_separable_state = choice(
                arange(0, n + 1), p=separable_states_weights
            )

            strategy = FixedLossSimulationStrategy(
                self.interferometer_matrix,
                number_of_particles_left_in_selected_separable_state,
                self.number_of_modes,
            )

            simulator = BosonSamplingSimulator(strategy)

            samples.append(simulator.get_classical_simulation_results(input_state)[0])

        return samples
