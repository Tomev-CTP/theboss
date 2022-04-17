__author__ = "Tomasz Rybotycki"

from numpy import complex128, delete, ndarray, vstack, zeros_like

from .network_simulation_strategy import NetworkSimulationStrategy
from ..boson_sampling_utilities.boson_sampling_utilities import (
    prepare_interferometer_matrix_in_expanded_space,
)


class LossyNetworkSimulationStrategy(NetworkSimulationStrategy):
    def __init__(self, matrix: ndarray) -> None:
        self._matrix = prepare_interferometer_matrix_in_expanded_space(matrix)

    def simulate(self, input_state: ndarray) -> ndarray:
        """
        This method is for simulating lossy network.

        Instead of using NxN matrix, what we need to do is create a 2N x 2N system, and then, depending on singular
        values of passed matrix, we have to transfer some photons into inaccessible, additional modes and then trim
        these additional modes.

        :param input_state: State before parsing through the interferometer. Assume mode occupation basis.
        :return: Lossy output state.
        """
        # Divide by two, because we have 2N x 2N matrix
        input_state = input_state.reshape(self._matrix.shape[0] // 2, 1)
        expansion_zeros = zeros_like(input_state, dtype=complex128)
        expanded_state = vstack([input_state, expansion_zeros])
        evolved_state = self._matrix @ expanded_state
        # Trim the resultant state
        while evolved_state.shape[0] > input_state.shape[0]:
            evolved_state = delete(evolved_state, evolved_state.shape[0] - 1)
        # Reshape to usual space and return.
        evolved_state = evolved_state.flatten()

        return evolved_state
