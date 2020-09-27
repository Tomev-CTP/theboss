from enum import IntEnum
from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration
from src.simulation_strategies.SimulationStrategy import SimulationStrategy
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy
from src.simulation_strategies.UniformLossSimulationStrategy import UniformLossSimulationStrategy
from src.simulation_strategies.GeneralizedCliffordsSimulationStrategy import GeneralizedCliffordsSimulationStrategy
from src.simulation_strategies.CliffordsRSimulationStrategy import CliffordsRSimulationStrategy


class StrategyTypes(IntEnum):
    FIXED_LOSS = 1
    UNIFORM_LOSS = 2
    CLIFFORD_R = 3
    GENERALIZED_CLIFFORD = 4


class SimulationStrategyFactory:
    def __init__(self, experiment_configuration: BosonSamplingExperimentConfiguration,
                 strategy_type: StrategyTypes = StrategyTypes.FIXED_LOSS):
        self._experiment_configuration = experiment_configuration
        self._strategy_type = strategy_type

    def generate_strategy(self) -> SimulationStrategy:
        """
            Generates simulation strategy of desired type. The type is selected in the constructor.
        :return: Simulation strategy of desired type.
        """
        if self._strategy_type == StrategyTypes.UNIFORM_LOSS:
            return self.__generate_uniform_losses_strategy()
        if self._strategy_type == StrategyTypes.CLIFFORD_R:
            return self.__generate_r_cliffords_strategy()
        if self._strategy_type == StrategyTypes.GENERALIZED_CLIFFORD:
            return self.__generate_generalized_cliffords_strategy()
        return self.__generate_fixed_losses_strategy()

    def __generate_fixed_losses_strategy(self):
        """
            Generates fixed loss strategy.
        :return: Fixed loss strategy.
        """
        return FixedLossSimulationStrategy(
            self._experiment_configuration.interferometer_matrix,
            self._experiment_configuration.number_of_particles_left,
            self._experiment_configuration.number_of_modes
        )

    def __generate_uniform_losses_strategy(self):
        """
            Generates uniform losses strategy.
        :return: Uniform losses strategy.
        """
        return UniformLossSimulationStrategy(
            self._experiment_configuration.interferometer_matrix,
            self._experiment_configuration.number_of_modes,
            self._experiment_configuration.probability_of_uniform_loss
        )

    def __generate_r_cliffords_strategy(self):
        """
            Generates Cliffords algorithm strategy using their code implemented in R.
        :return: Cliffords strategy in R.
        """
        return CliffordsRSimulationStrategy(
            self._experiment_configuration.number_of_particles_left,
            self._experiment_configuration.interferometer_matrix
        )

    def __generate_generalized_cliffords_strategy(self):
        """
            Generates generalized Cliffords strategy from Oszmaniec / Brod.
        :return: Generalized Cliffords strategy.
        """
        return GeneralizedCliffordsSimulationStrategy(
            self._experiment_configuration.interferometer_matrix
        )

