__author__ = "Tomasz Rybotycki"

from numpy import delete, ndarray, vstack, zeros_like

from src.boson_sampling_utilities.Boson_Sampling_Utilities import prepare_interferometer_matrix_in_expanded_space
from src.boson_sampling_utilities.permanent_calculators.BSPermanentCalculatorInterface import \
    BSPermanentCalculatorInterface
from src.simulation_strategies.GeneralizedCliffordsSimulationStrategy import \
    GeneralizedCliffordsSimulationStrategy
from src.simulation_strategies.SimulationStrategyInterface import SimulationStrategyInterface


class LossyNetworksGeneralizedCliffordsSimulationStrategy(SimulationStrategyInterface):
    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        bs_permanent_calculator.matrix = prepare_interferometer_matrix_in_expanded_space(bs_permanent_calculator.matrix)
        self._helper_strategy = GeneralizedCliffordsSimulationStrategy(bs_permanent_calculator)

    def simulate(self, input_state: ndarray, samples_number: int = 1) -> [ndarray]:
        input_state = input_state.reshape(len(input_state), )  # Divide by two, coz we have 2N x 2N matrix
        expansion_zeros = zeros_like(input_state)
        expanded_state = vstack([input_state, expansion_zeros])
        expanded_state = expanded_state.reshape(2 * len(input_state), )
        expanded_samples = self._helper_strategy.simulate(expanded_state, samples_number)
        # Trim output state
        samples = []

        for output_state in expanded_samples:
            while len(output_state) > len(input_state):
                output_state = delete(output_state, len(output_state) - 1)
            samples.append(output_state)

        return samples
