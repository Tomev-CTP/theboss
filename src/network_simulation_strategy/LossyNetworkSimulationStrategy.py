from numpy import block, delete, diag, identity, ndarray, sqrt, vstack, zeros_like
from numpy.linalg import svd

from src.network_simulation_strategy.NetworkSimulationStrategy import NetworkSimulationStrategy


class LossyNetworkSimulationStrategy(NetworkSimulationStrategy):
    def __init__(self, matrix: ndarray):
        self._matrix = matrix

    def simulate(self, input_state: ndarray) -> ndarray:
        """
        This method is for simulating lossy network. Instead of using NxN matrix, what we need to do is create a
        2N x 2N system, and then, depending on singular values of passed matrix, we have to transfer some photons
        into inaccessible modes. This
        :param input_state: State before parsing through the interferometer. Assume mode occupation basis.
        :return: Lossy output state.
        """
        input_state = input_state.reshape(self._matrix.shape[0], 1)
        # Decompose the matrix first.
        u_matrix, singular_values_vector, v_matrix = svd(self._matrix)
        # Then apply U.
        u_evolved_state = u_matrix @ input_state
        # Then an expansion gotta happen.
        expansion_zeros = zeros_like(u_evolved_state)
        expanded_state = vstack([u_evolved_state, expansion_zeros])
        # We have to expand the matrices as well.
        zeros_matrix = zeros_like(self._matrix)
        identity_expansion = identity(zeros_matrix.shape[0])
        singular_values_matrix_expansion = self.__prepare_singular_values_matrix_expansion(singular_values_vector)
        expanded_v_matrix = block([[v_matrix, zeros_matrix], [zeros_matrix, identity_expansion]])
        singular_values_matrix = diag(singular_values_vector)
        expanded_singular_values_matrix = \
            block([[singular_values_matrix, zeros_matrix], [singular_values_matrix_expansion, zeros_matrix]])
        # Proceed with the rest of the evolution.
        evolved_state = expanded_singular_values_matrix @ expanded_state
        evolved_state = expanded_v_matrix @ evolved_state
        # Trim the resultant state
        while evolved_state.shape[0] > input_state.shape[0]:
            evolved_state = delete(evolved_state, evolved_state.shape[0] - 1)
        # Reshape to usual states and result.
        evolved_state = evolved_state.flatten()

        return evolved_state

    @staticmethod
    def __prepare_singular_values_matrix_expansion(singular_values_vector: ndarray) -> ndarray:
        """
        Prepares matrix expansion for singular values matrix. In order to simulate lossy interferometer we need to
        work on hilbert space with greater dimension. In that case we need to expand our operators. Singular values
        matrix should be build in such way, that if (s_1, ..., s_n) are singular values, the expansion is then diagonal
        matrix with values [sqrt(1 - s_1^2, ..., sqrt(1- s^2_n)].

        Note, that singular values for lossless nets will be given by an unitary matrix, and this will have singular
        values equal to 1, which corresponds to expansion matrix being 0.

        :param singular_values_vector: Vector with singular values of given matrix.
        :return:
        """
        expansion_values = []
        for singular_value in singular_values_vector:
            expansion_values.append(sqrt(1.0 - pow(singular_value, 2)))
        return diag(expansion_values)
