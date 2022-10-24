__author__ = "Tomasz Rybotycki"

"""
    This file holds an implementation of generic boson sampling experiment simulator.
"""

# TODO TR: Expand the simulator to justify it's existence.
#   - Add frequencies counting
#   - ???

from typing import List, Sequence, Tuple

from .simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)


class BosonSamplingSimulator:
    """
    An implementation of the BosonSampling simulator. It uses the strategies from
    the simulation_strategies module,
    """

    def __init__(self, simulation_strategy: SimulationStrategyInterface) -> None:
        self._simulation_strategy = simulation_strategy

    def get_classical_simulation_results(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int]]:
        """
        A method that performs the classical simulation, by calling the simulate method
        of the strategy.

        :param input_state:
            An input (Fock) state used in the simulation.
        :param samples_number:
            The number of samples to sample.

        :return:
            The specified number of samples.
        """
        return self._simulation_strategy.simulate(input_state, samples_number)
