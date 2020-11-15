import enum

from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration
from src.simulation_strategies.CliffordsRSimulationStrategy import CliffordsRSimulationStrategy
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy
from src.simulation_strategies.GeneralizedCliffordsSimulationStrategy import GeneralizedCliffordsSimulationStrategy
from src.simulation_strategies.LossyNetworksGeneralizedCliffordsSimulationStrategy import \
    LossyNetworksGeneralizedCliffordsSimulationStrategy
from src.simulation_strategies.SimulationStrategy import SimulationStrategy
from src.simulation_strategies.UniformLossSimulationStrategy import UniformLossSimulationStrategy


class StrategyTypes(enum.IntEnum):
    FIXED_LOSS = enum.auto()
    UNIFORM_LOSS = enum.auto()
    CLIFFORD_R = enum.auto()
    GENERALIZED_CLIFFORD = enum.auto()
    LOSSY_NET_GENERALIZED_CLIFFORD = enum.auto()


class SimulationStrategyFactory:
    def __init__(self, experiment_configuration: BosonSamplingExperimentConfiguration,
                 strategy_type: StrategyTypes = StrategyTypes.FIXED_LOSS) -> None:
        self._experiment_configuration = experiment_configuration
        self._strategy_type = strategy_type
        self._strategy_mapping = {
            StrategyTypes.FIXED_LOSS: self.__generate_fixed_losses_strategy,
            StrategyTypes.UNIFORM_LOSS: self.__generate_uniform_losses_strategy,
            StrategyTypes.CLIFFORD_R: self.__generate_r_cliffords_strategy,
            StrategyTypes.GENERALIZED_CLIFFORD: self.__generate_generalized_cliffords_strategy,
            StrategyTypes.LOSSY_NET_GENERALIZED_CLIFFORD: self.__generate_lossy_net_generalized_cliffords_strategy
        }

    def set_strategy_type(self, strategy_type: StrategyTypes) -> None:
        self._strategy_type = strategy_type

    def set_experiment_configuration(self, experiment_configuration: BosonSamplingExperimentConfiguration) -> None:
        self._experiment_configuration = experiment_configuration

    def generate_strategy(self) -> SimulationStrategy:
        """
            Generates simulation strategy of desired type. The type is selected in the constructor.
        :return: Simulation strategy of desired type.
        """
        handler = self._strategy_mapping.get(self._strategy_type, self.__generate_fixed_losses_strategy)
        return handler()

    def __generate_fixed_losses_strategy(self) -> FixedLossSimulationStrategy:
        """
            Generates fixed loss strategy.
        :return: Fixed loss strategy.
        """
        return FixedLossSimulationStrategy(
            interferometer_matrix=self._experiment_configuration.interferometer_matrix,
            number_of_photons_left=self._experiment_configuration.number_of_particles_left,
            number_of_observed_modes=self._experiment_configuration.number_of_modes,
            network_simulation_strategy=self._experiment_configuration.network_simulation_strategy
        )

    def __generate_uniform_losses_strategy(self) -> UniformLossSimulationStrategy:
        """
            Generates uniform losses strategy.
        :return: Uniform losses strategy.
        """
        return UniformLossSimulationStrategy(
            self._experiment_configuration.interferometer_matrix,
            self._experiment_configuration.number_of_modes,
            self._experiment_configuration.probability_of_uniform_loss
        )

    def __generate_r_cliffords_strategy(self) -> CliffordsRSimulationStrategy:
        """
            Generates Cliffords algorithm strategy using their code implemented in R.
        :return: Cliffords strategy in R.
        """
        return CliffordsRSimulationStrategy(self._experiment_configuration.interferometer_matrix)

    def __generate_generalized_cliffords_strategy(self) -> GeneralizedCliffordsSimulationStrategy:
        """
            Generates generalized Cliffords strategy from Oszmaniec / Brod.
        :return: Generalized Cliffords strategy.
        """
        return GeneralizedCliffordsSimulationStrategy(
            self._experiment_configuration.interferometer_matrix
        )

    def __generate_lossy_net_generalized_cliffords_strategy(self) \
            -> LossyNetworksGeneralizedCliffordsSimulationStrategy:
        """
            Generates generalized Cliffords strategy for lossy networks from Oszmaniec / Brod 2020.
        :return: Generalized Cliffords strategy for lossy networks.
        """
        return LossyNetworksGeneralizedCliffordsSimulationStrategy(
            self._experiment_configuration.interferometer_matrix
        )
