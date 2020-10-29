from numpy import block, delete, diag, identity, ndarray, sqrt, vstack, zeros_like, array
from numpy.linalg import svd

from src.network_simulation_strategy.NetworkSimulationStrategy import NetworkSimulationStrategy
from src.Boson_Sampling_Utilities import prepare_interferometer_matrix_in_expanded_space

class LossyNetworkSimulationStrategy(NetworkSimulationStrategy):
    def __init__(self, matrix: ndarray):
        self._matrix = prepare_interferometer_matrix_in_expanded_space(matrix)

    def simulate(self, input_state: ndarray) -> ndarray:
        """
        This method is for simulating lossy network. Instead of using NxN matrix, what we need to do is create a
        2N x 2N system, and then, depending on singular values of passed matrix, we have to transfer some photons
        into inaccessible, additional modes and then trim these additional modes.
        :param input_state: State before parsing through the interferometer. Assume mode occupation basis.
        :return: Lossy output state.
        """
        input_state = input_state.reshape(int(self._matrix.shape[0] / 2), 1)  # Divide by two, coz we have 2N x 2N matrix
        expansion_zeros = zeros_like(input_state)
        expanded_state = vstack([input_state, expansion_zeros])
        evolved_state = self._matrix @ expanded_state
        # Trim the resultant state
        while evolved_state.shape[0] > input_state.shape[0]:
            evolved_state = delete(evolved_state, evolved_state.shape[0] - 1)
        # Reshape to usual states and result.
        evolved_state = evolved_state.flatten()

        return evolved_state
