__author__ = "Tomasz Rybotycki"

from typing import List, Sequence, Tuple

from theboss.math_utilities import choice

from theboss.simulation_strategies.mean_field_fixed_loss_simulation_strategy import (
    MeanFieldFixedLossSimulationStrategy,
)
from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)
from theboss.boson_sampling_utilities import compute_binomial_weights


class UniformLossSimulationStrategy(SimulationStrategyInterface):
    """
    An implementation of a mean-field simulation strategy for BS experiments with
    uniform losses.
    """

    def __init__(
        self,
        interferometer_matrix: Sequence[Sequence[complex]],
        number_of_modes: int,
        transmission_probability: float,
    ) -> None:
        self.interferometer_matrix: Sequence[Sequence[complex]] = interferometer_matrix
        self.number_of_modes: int = number_of_modes
        self.transmission_probability: float = transmission_probability

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Returns a list of samples drawn from the uniformly lossy mean-field
        distribution.

        :param input_state:
            A Fock state at the input of the interferometer.
        :param samples_number:
            The number of samples to draw from the uniformly lossy mean-field
            distribution.

        :return:
            A list of Fock states sampled from the uniformly lossy mean-field
            distribution.
        """
        initial_number_of_particles = int(sum(input_state))

        possible_number_of_particles: List[int] = [
            i for i in range(initial_number_of_particles + 1)
        ]
        separable_states_weights: List[float] = compute_binomial_weights(
            initial_number_of_particles, self.transmission_probability
        )

        samples: List[Tuple[int, ...]] = []

        while len(samples) < samples_number:

            number_of_particles_left_in_selected_separable_state = choice(
                possible_number_of_particles, separable_states_weights
            )

            strategy = MeanFieldFixedLossSimulationStrategy(
                self.interferometer_matrix,
                number_of_particles_left_in_selected_separable_state,
                self.number_of_modes,
            )

            samples.append(strategy.simulate(input_state, 1)[0])

        return samples
