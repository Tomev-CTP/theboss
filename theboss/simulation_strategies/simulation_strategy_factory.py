__author__ = "Tomasz Rybotycki"

"""
    This file holds a factory for simulation strategies.
"""

import enum
from copy import deepcopy

from .fixed_loss_simulation_strategy import FixedLossSimulationStrategy
from .generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
)
from .generalized_cliffords_uniform_losses_simulation_strategy import (
    GeneralizedCliffordsUniformLossesSimulationStrategy,
)
from .lossy_networks_generalized_cliffords_simulation_strategy import (
    LossyNetworksGeneralizedCliffordsSimulationStrategy,
)
from .lossy_state_approximated_simulation_strategy import (
    LossyStateApproximationSimulationStrategy,
)
from .nonuniform_losses_approximation_strategy import (
    NonuniformLossesApproximationStrategy,
)
from .simulation_strategy_interface import SimulationStrategyInterface
from .uniform_loss_simulation_strategy import UniformLossSimulationStrategy
from ..boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)
from ..distribution_calculators.bs_distribution_calculator_interface import (
    BosonSamplingExperimentConfiguration,
)


class StrategyType(enum.IntEnum):
    FIXED_LOSS = enum.auto()
    UNIFORM_LOSS = enum.auto()
    CLIFFORD_R = enum.auto()
    GCC = enum.auto()
    LOSSY_NET_GCC = enum.auto()
    LOSSLESS_MODES_STRATEGY = enum.auto()
    UNIFORM_LOSSES_GCC = enum.auto()
    BOBS = enum.auto()  # Brod-Oszmaniec BosonSampling
    UNIFORM_LOSSES_BOBS = enum.auto()


class SimulationStrategyFactory:
    def __init__(
        self,
        experiment_configuration: BosonSamplingExperimentConfiguration,
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        strategy_type: StrategyType = StrategyType.FIXED_LOSS,
    ) -> None:

        self._experiment_configuration = experiment_configuration
        self._strategy_type = strategy_type
        self._bs_permanent_calculator = deepcopy(bs_permanent_calculator)

        self._strategy_mapping = {
            StrategyType.FIXED_LOSS: self._generate_fixed_losses_strategy,
            StrategyType.UNIFORM_LOSS: self._generate_uniform_losses_strategy,
            StrategyType.CLIFFORD_R: self._generate_r_cliffords_strategy,
            StrategyType.GCC: self._generate_gcc_strategy,
            StrategyType.LOSSY_NET_GCC: self._generate_lossy_net_gcc_strategy,
            StrategyType.UNIFORM_LOSSES_GCC: self._generate_uniform_losses_gcc_strategy,
            StrategyType.BOBS: self._generate_bobs_strategy,
            StrategyType.UNIFORM_LOSSES_BOBS: self._generate_uniform_losses_bobs_strategy,
        }

        self._available_threads_number = -1

    @property
    def available_threads_number(self) -> int:
        return self._available_threads_number

    @available_threads_number.setter
    def available_threads_number(self, available_threads_number: int) -> None:
        self._available_threads_number = available_threads_number

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
    def experiment_configuration(
        self, experiment_configuration: BosonSamplingExperimentConfiguration
    ) -> None:
        self._experiment_configuration = experiment_configuration

    @property
    def bs_permanent_calculator(self) -> BSPermanentCalculatorInterface:
        return self._bs_permanent_calculator

    @bs_permanent_calculator.setter
    def bs_permanent_calculator(
        self, bs_permanent_calculator: BSPermanentCalculatorInterface
    ) -> None:
        self._bs_permanent_calculator = deepcopy(bs_permanent_calculator)

    def generate_strategy(self) -> SimulationStrategyInterface:
        """
        Generates simulation strategy of desired type. The type is selected in the
        constructor.

        :return:    Simulation strategy of desired type.
        """
        handler = self._strategy_mapping.get(
            self._strategy_type, self._generate_fixed_losses_strategy
        )
        return handler()

    def _generate_fixed_losses_strategy(self) -> FixedLossSimulationStrategy:
        """
        Generates fixed losses' strategy from [1].

        :return: Fixed loss strategy.
        """
        return FixedLossSimulationStrategy(
            interferometer_matrix=self._experiment_configuration.interferometer_matrix,
            number_of_photons_left=self._experiment_configuration.number_of_particles_left,
            number_of_observed_modes=self._experiment_configuration.number_of_modes,
            network_simulation_strategy=self._experiment_configuration.network_simulation_strategy,
        )

    def _generate_uniform_losses_strategy(self) -> UniformLossSimulationStrategy:
        """
        Generates uniform losses' strategy from [1].

        :return: Uniform losses strategy.
        """
        return UniformLossSimulationStrategy(
            self._experiment_configuration.interferometer_matrix,
            self._experiment_configuration.number_of_modes,
            self._experiment_configuration.uniform_transmissivity,
        )

    def _generate_r_cliffords_strategy(self):
        """
        Generates Cliffords algorithm strategy using their code implemented in R.

        :return: Cliffords strategy in R.
        """
        try:
            from rpy2.robjects import packages
        except ImportError as error:
            raise ImportError(
                f"{str(error)}. You have to install 'rpy2' in order to use the"
                "'CLIFFORD_R' strategy."
            )

        required_packages = ("BosonSampling", "Rcpp", "RcppArmadillo")

        missing_packages = [
            package
            for package in required_packages
            if not packages.isinstalled(package)
        ]

        if missing_packages:
            raise ImportError(
                f"Some R packages are missing: missing_packages='{required_packages}'"
            )

        from .cliffords_r_simulation_strategy import CliffordsRSimulationStrategy

        return CliffordsRSimulationStrategy(
            self._experiment_configuration.interferometer_matrix
        )

    def _generate_gcc_strategy(self) -> GeneralizedCliffordsSimulationStrategy:
        """
        Generates generalized Cliffords strategy from Oszmaniec & Brod.

        :return: Generalized Cliffords strategy.
        """
        return GeneralizedCliffordsSimulationStrategy(
            deepcopy(self.bs_permanent_calculator)
        )

    def _generate_lossy_net_gcc_strategy(
        self,
    ) -> LossyNetworksGeneralizedCliffordsSimulationStrategy:
        """
        Generates generalized Cliffords strategy for lossy networks from [2].

        :return: Generalized Cliffords strategy for lossy networks.
        """
        return LossyNetworksGeneralizedCliffordsSimulationStrategy(
            deepcopy(self.bs_permanent_calculator)
        )

    def _generate_uniform_losses_gcc_strategy(
        self,
    ) -> GeneralizedCliffordsUniformLossesSimulationStrategy:
        return GeneralizedCliffordsUniformLossesSimulationStrategy(
            deepcopy(self.bs_permanent_calculator),
            self._experiment_configuration.uniform_transmissivity,
        )

    def _generate_bobs_strategy(self) -> NonuniformLossesApproximationStrategy:
        approximated_modes_number = self._experiment_configuration.number_of_modes
        approximated_modes_number -= self._experiment_configuration.hierarchy_level

        return NonuniformLossesApproximationStrategy(
            bs_permanent_calculator=deepcopy(self.bs_permanent_calculator),
            approximated_modes_number=approximated_modes_number,
            modes_transmissivity=self._experiment_configuration.uniform_transmissivity,
        )

    def _generate_uniform_losses_bobs_strategy(
        self,
    ) -> LossyStateApproximationSimulationStrategy:
        return LossyStateApproximationSimulationStrategy(
            bs_permanent_calculator=self._bs_permanent_calculator,
            uniform_transmissivity=self._experiment_configuration.uniform_transmissivity,
            hierarchy_level=self._experiment_configuration.hierarchy_level,
            threads_number=self._available_threads_number,
        )
