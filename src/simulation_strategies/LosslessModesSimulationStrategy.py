from typing import Optional, List

from numpy import ndarray, zeros, sqrt

from src.network_simulation_strategy.NetworkSimulationStrategy import NetworkSimulationStrategy
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy


class LosslessModesSimulationStrategy(FixedLossSimulationStrategy):
    """
        This class implements an upgraded version of approximation algorithm presented in [1]. It's
        described in more detail in [2].
    """

    def __init__(self, interferometer_matrix: ndarray,
                 number_of_photons_left: int, number_of_observed_modes: int, number_of_lossless_modes: int = 0,
                 network_simulation_strategy: Optional[NetworkSimulationStrategy] = None) \
            -> None:
        self.number_of_lossless_modes = number_of_lossless_modes
        super().__init__(interferometer_matrix, number_of_photons_left, number_of_observed_modes,
                         network_simulation_strategy)

    def _prepare_initial_state(self, input_state: ndarray) -> ndarray:
        """
            This method prepares initial state for algorithm, as described in [2]. It'd expect that the
            input state would be like [1, 1, 1, ..., 1 , n, 0, 0, ..., 0], but in the code I'll make
            no such assumption. The 1 here would only occupy lossless modes, and n particles would be
            on one of the lossy modes.
            :param input_state: Initial lossy bosonic state.
            :return: Returns initial state of the formula, which is 1 particle on all lossless modes and then
            an equal superposition of n photons 'smeared' on the next n modes.
        """
        initial_number_of_photons = int(sum(input_state))
        number_of_photons_in_lossy_modes = initial_number_of_photons - self.number_of_lossless_modes
        prepared_state = zeros(self.number_of_observed_modes, dtype=float)

        # TR TODO: Do note that initial number of photons can possibly exceed the number of modes and I have no\
        #  knowledge yet how to proceed in such case. For now I assume m ~ n^2, which is the usual assumption in BS.
        for i in range(self.number_of_lossless_modes, initial_number_of_photons):
            prepared_state[i] = 1. / sqrt(number_of_photons_in_lossy_modes)

        prepared_state = FixedLossSimulationStrategy._randomize_modes_phases(prepared_state)

        #
        for i in range(self.number_of_lossless_modes):
            prepared_state[i] = 1.

        return prepared_state
