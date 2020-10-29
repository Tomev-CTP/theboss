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

    @staticmethod
    def __prepare_singular_values_matrix_expansion(singular_values_vector: array) -> ndarray:
        """
        Prepares matrix expansion for singular values matrix. In order to simulate lossy interferometer we need to
        work on hilbert space with greater dimension. In that case we need to expand our operators. Singular values
        matrix should be build in such way, that if (s_1, ..., s_n) are singular values, the expansion is then diagonal
        matrix with values [sqrt(1 - s_1^2, ..., sqrt(1- s_n^n)].

        Note, that singular values for lossless nets will be given by an unitary matrix, and this will have singular
        values equal to 1, which corresponds to expansion matrix being 0.

        :param singular_values_vector: Vector with singular values of given matrix.
        :return: A matrix expansion for inaccessible modes of expanded space.
        """
        expansion_values = []
        for singular_value in singular_values_vector:
            expansion_values.append(sqrt(1.0 - pow(singular_value, 2)))
        return diag(expansion_values)
