from src.network_simulation_strategy import NetworkSimulationStrategy
from numpy import ndarray, zeros_like, vstack, block, identity, diag, delete
from numpy.linalg import svd


class LossyNetworkSimulationStrategy(NetworkSimulationStrategy):
    def __init__(self):
        pass

    @staticmethod
    def simulate(matrix: ndarray, input_state: ndarray):
        input_state = input_state.reshape(matrix.shape[0], 1)
        # Decompose the matrix first.
        u_matrix, singular_values_vector, v_matrix = \
            svd(matrix)
        # Then apply U.
        u_evolved_state = u_matrix @ input_state
        # Then an expansion gotta happen.
        expansion_zeros = zeros_like(u_evolved_state)
        expanded_state = vstack([u_evolved_state, expansion_zeros])
        # We have to expand the matrices as well.
        zeros_matrix = zeros_like(matrix)
        identity_expansion = identity(zeros_matrix.shape[0])
        expanded_v_matrix = block([[v_matrix, zeros_matrix], [zeros_matrix, identity_expansion]])
        singular_values_matrix = diag(singular_values_vector)
        expanded_singular_values_matrix = \
            block([[singular_values_matrix, zeros_matrix], [zeros_matrix, zeros_matrix]])
        for i in range(matrix.shape[0]):
            expanded_singular_values_matrix[i + matrix.shape[0]][i] = \
                1 - expanded_singular_values_matrix[i][i]
        # Proceed with the rest of the evolution.
        evolved_state = singular_values_matrix @ u_evolved_state
        evolved_state = expanded_v_matrix @ evolved_state
        # Trim the resultant state
        while evolved_state.shape[0] > input_state.shape[0]:
            evolved_state = delete(evolved_state, evolved_state.shape[0] - 1)
        # Reshape to usual states and result.
        evolved_state = evolved_state.reshape(1, )
        return evolved_state
