__author__ = "Tomasz Rybotycki"

"""
    This file holds a factory for simulation strategies.
"""

import enum
from copy import deepcopy

from .cliffords_r_simulation_strategy import CliffordsRSimulationStrategy
from .fixed_loss_simulation_strategy import FixedLossSimulationStrategy
from .generalized_cliffords_simulation_strategy import GeneralizedCliffordsSimulationStrategy
from .generalized_cliffords_uniform_losses_simulation_strategy import \
    GeneralizedCliffordsUniformLossesSimulationStrategy
from .lossy_networks_generalized_cliffords_simulation_strategy import \
    LossyNetworksGeneralizedCliffordsSimulationStrategy
from .lossy_state_approximated_simulation_strategy import LossyStateApproximationSimulationStrategy
from .nonuniform_losses_approximation_strategy import NonuniformLossesApproximationStrategy
from .simulation_strategy_interface import SimulationStrategyInterface
from .uniform_loss_simulation_strategy import UniformLossSimulationStrategy
from ..boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import \
    BSPermanentCalculatorInterface
from ..distribution_calculators.bs_distribution_calculator_interface import BosonSamplingExperimentConfiguration


class StrategyType(enum.IntEnum):
    FIXED_LOSS = enum.auto()
    UNIFORM_LOSS = enum.auto()
    CLIFFORD_R = enum.auto()
    GENERALIZED_CLIFFORD = enum.auto()
    LOSSY_NET_GENERALIZED_CLIFFORD = enum.auto()
    LOSSLESS_MODES_STRATEGY = enum.auto()
    GENERALIZED_U_LOSSY_CLIFFORD = enum.auto()
    NONUNIFORM_APPROXIMATED = enum.auto()
    UNIFORM_LOSSY_STATE_APPROXIMATED = enum.auto()


class SimulationStrategyFactory:
    def __init__(self, experiment_configuration: BosonSamplingExperimentConfiguration,
                 bs_permanent_calculator: BSPermanentCalculatorInterface,
                 strategy_type: StrategyType = StrategyType.FIXED_LOSS) -> None:
        self._experiment_configuration = experiment_configuration
        self._strategy_type = strategy_type
        self._bs_permanent_calculator = deepcopy(bs_permanent_calculator)
        self._strategy_mapping = {
            StrategyType.FIXED_LOSS: self._generate_fixed_losses_strategy,
            StrategyType.UNIFORM_LOSS: self._generate_uniform_losses_strategy,
            StrategyType.CLIFFORD_R: self._generate_r_cliffords_strategy,
            StrategyType.GENERALIZED_CLIFFORD: self._generate_generalized_cliffords_strategy,
            StrategyType.LOSSY_NET_GENERALIZED_CLIFFORD: self._generate_lossy_net_generalized_cliffords_strategy,
            StrategyType.GENERALIZED_U_LOSSY_CLIFFORD: self._generate_u_lossy_generalized_cliffords_strategy,
            StrategyType.NONUNIFORM_APPROXIMATED: self._generate_nonuniform_approximating_strategy,
            StrategyType.UNIFORM_LOSSY_STATE_APPROXIMATED: self._generate_uniform_lossy_state_approximating_strategy
        }

    @property
    def strategy_type(self) -> StrategyType:
        return self._strategy_type

    @strategy_type.setter
    def strategy_type(self, strategy_type: StrategyType) -> None:
        self._strategy_type = strategy_type

    @property
    def experiment_configuration(self) -> BosonSamplingExperimentConfiguration:
        return self._experiment_configuration

    @experiment_configuration.setter
    def experiment_configuration(self, experiment_configuration: BosonSamplingExperimentConfiguration) -> None:
        self._experiment_configuration = experiment_configuration

    @property
    def bs_permanent_calculator(self) -> BSPermanentCalculatorInterface:
        return self._bs_permanent_calculator

    @bs_permanent_calculator.setter
    def bs_permanent_calculator(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        self._bs_permanent_calculator = deepcopy(bs_permanent_calculator)

    def generate_strategy(self) -> SimulationStrategyInterface:
        """
            Generates simulation strategy of desired type. The type is selected in the constructor.
        :return: Simulation strategy of desired type.
        """
        handler = self._strategy_mapping.get(self._strategy_type, self._generate_fixed_losses_strategy)
        return handler()

    def _generate_fixed_losses_strategy(self) -> FixedLossSimulationStrategy:
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

    def _generate_uniform_losses_strategy(self) -> UniformLossSimulationStrategy:
        """
            Generates uniform losses strategy.
        :return: Uniform losses strategy.
        """
        return UniformLossSimulationStrategy(
            self._experiment_configuration.interferometer_matrix,
            self._experiment_configuration.number_of_modes,
            self._experiment_configuration.uniform_transmissivity
        )

    def _generate_r_cliffords_strategy(self) -> CliffordsRSimulationStrategy:
        """
            Generates Cliffords algorithm strategy using their code implemented in R.
        :return: Cliffords strategy in R.
        """
        return CliffordsRSimulationStrategy(self._experiment_configuration.interferometer_matrix)

    def _generate_generalized_cliffords_strategy(self) -> GeneralizedCliffordsSimulationStrategy:
        """
            Generates generalized Cliffords strategy from Oszmaniec / Brod.
        :return: Generalized Cliffords strategy.
        """
        return GeneralizedCliffordsSimulationStrategy(deepcopy(self.bs_permanent_calculator))

    def _generate_lossy_net_generalized_cliffords_strategy(self) \
            -> LossyNetworksGeneralizedCliffordsSimulationStrategy:
        """
            Generates generalized Cliffords strategy for lossy networks from Oszmaniec / Brod 2020.
        :return: Generalized Cliffords strategy for lossy networks.
        """
        return LossyNetworksGeneralizedCliffordsSimulationStrategy(deepcopy(self.bs_permanent_calculator))

    def _generate_u_lossy_generalized_cliffords_strategy(self) -> GeneralizedCliffordsUniformLossesSimulationStrategy:
        return GeneralizedCliffordsUniformLossesSimulationStrategy(deepcopy(self.bs_permanent_calculator),
                                                                   self._experiment_configuration.uniform_transmissivity)

    def _generate_nonuniform_approximating_strategy(self) -> NonuniformLossesApproximationStrategy:
        return NonuniformLossesApproximationStrategy(deepcopy(self.bs_permanent_calculator),
                                                     self._experiment_configuration.number_of_modes - self._experiment_configuration.hierarchy_level,
                                                     self._experiment_configuration.uniform_transmissivity)

    def _generate_uniform_lossy_state_approximating_strategy(self) -> LossyStateApproximationSimulationStrategy:
        return LossyStateApproximationSimulationStrategy(
            bs_permanent_calculator=self._bs_permanent_calculator,
            uniform_transmissivity=self._experiment_configuration.uniform_transmissivity,
            hierarchy_level=self._experiment_configuration.hierarchy_level
        )
