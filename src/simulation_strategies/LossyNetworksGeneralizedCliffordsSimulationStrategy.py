__author__ = 'Tomasz Rybotycki'

from numpy import array, delete, ndarray, vstack, zeros_like

from src.Boson_Sampling_Utilities import prepare_interferometer_matrix_in_expanded_space
from src.simulation_strategies.GeneralizedCliffordsSimulationStrategy import GeneralizedCliffordsSimulationStrategy
from src.simulation_strategies.SimulationStrategy import SimulationStrategy


class LossyNetworksGeneralizedCliffordsSimulationStrategy(SimulationStrategy):
    def __init__(self, interferometer_matrix: ndarray) -> None:
        self._helper_strategy = GeneralizedCliffordsSimulationStrategy(
            prepare_interferometer_matrix_in_expanded_space(interferometer_matrix))

    def simulate(self, input_state: ndarray) -> ndarray:
        input_state = input_state.reshape(len(input_state), )  # Divide by two, coz we have 2N x 2N matrix
        expansion_zeros = zeros_like(input_state)
        expanded_state = vstack([input_state, expansion_zeros])
        expanded_state = expanded_state.reshape(2 * len(input_state), )
        output_state = self._helper_strategy.simulate(expanded_state)
        # Trim output state
        while len(output_state) > len(input_state):
            output_state = delete(output_state, len(output_state) - 1)
        return array(output_state)
