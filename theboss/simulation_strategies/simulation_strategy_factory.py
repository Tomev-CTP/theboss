__author__ = "Tomasz Rybotycki"

"""
    This file holds a factory for simulation strategies.
"""

import enum
from copy import deepcopy

from .mean_field_fixed_loss_simulation_strategy import (
    MeanFieldFixedLossSimulationStrategy,
)
from .generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
)
from .generalized_cliffords_uniform_losses_simulation_strategy import (
    GeneralizedCliffordsUniformLossesSimulationStrategy,
)
from .generalized_cliffords_nonuniform_losses_simulation_strategy import (
    GeneralizedCliffordsNonuniformLossesSimulationStrategy,
)
from .generalized_mean_field_uniform_losses_simulation_strategy import (
    GeneralizedMeanFieldUniformLossesSimulationStrategy,
)
from .generalized_mean_field_nonuniform_losses_simulation_strategy import (
    GeneralizedMeanFieldNonuniformLossesSimulationStrategy,
)
from .simulation_strategy_interface import SimulationStrategyInterface
from .mean_field_uniform_losses_simulation_strategy import UniformLossSimulationStrategy
from theboss.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)
from ..distribution_calculators.bs_distribution_calculator_interface import (
    BosonSamplingExperimentConfiguration,
)


class StrategyType(enum.IntEnum):
    MF_FIXED_LOSS = enum.auto()
    MF_UNIFORM_LOSS = enum.auto()
    CLIFFORD_R = enum.auto()
    GCC = enum.auto()
    GCC_GENERAL_LOSSES = enum.auto()
    GCC_UNIFORM_LOSSES = enum.auto()
    GMF_GENERAL_LOSSES = enum.auto()
    GMF_UNIFORM_LOSSES = enum.auto()


class SimulationStrategyFactory:
    def __init__(
        self,
        experiment_configuration: BosonSamplingExperimentConfiguration,
        bs_permanent_calculator: BSPermanentCalculatorInterface,
        strategy_type: StrategyType = StrategyType.MF_FIXED_LOSS,
    ) -> None:

        self._experiment_configuration = experiment_configuration
        self._strategy_type = strategy_type
        self._bs_permanent_calculator = deepcopy(bs_permanent_calculator)

        self._strategy_mapping = {
            StrategyType.MF_FIXED_LOSS: self._generate_mf_fixed_losses_strategy,
            StrategyType.MF_UNIFORM_LOSS: self._generate_mf_uniform_losses_strategy,
            StrategyType.CLIFFORD_R: self._generate_r_cliffords_strategy,
            StrategyType.GCC: self._generate_gcc_strategy,
            StrategyType.GCC_GENERAL_LOSSES: self._generate_gcc_general_losses_strategy,
            StrategyType.GCC_UNIFORM_LOSSES: self._generate_gcc_uniform_losses_strategy,
            StrategyType.GMF_GENERAL_LOSSES: self._generate_gmf_general_losses_strategy,
            StrategyType.GMF_UNIFORM_LOSSES: self._generate_gmf_uniform_losses_strategy,
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
            self._strategy_type, self._generate_mf_fixed_losses_strategy
        )
        return handler()

    def _generate_mf_fixed_losses_strategy(
        self,
    ) -> MeanFieldFixedLossSimulationStrategy:
        """
        Generates fixed losses' strategy from [1].

        :return: Fixed loss strategy.
        """
        return MeanFieldFixedLossSimulationStrategy(
            interferometer_matrix=self._experiment_configuration.interferometer_matrix,
            number_of_photons_left=self._experiment_configuration.number_of_particles_left,
            number_of_observed_modes=self._experiment_configuration.number_of_modes,
            network_simulation_strategy=self._experiment_configuration.network_simulation_strategy,
        )

    def _generate_mf_uniform_losses_strategy(self) -> UniformLossSimulationStrategy:
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

    def _generate_gcc_general_losses_strategy(
        self,
    ) -> GeneralizedCliffordsNonuniformLossesSimulationStrategy:
        """
        Generates generalized Cliffords strategy for lossy networks from [2].

        :return: Generalized Cliffords strategy for lossy networks.
        """
        return GeneralizedCliffordsNonuniformLossesSimulationStrategy(
            deepcopy(self.bs_permanent_calculator)
        )

    def _generate_gcc_uniform_losses_strategy(
        self,
    ) -> GeneralizedCliffordsUniformLossesSimulationStrategy:
        return GeneralizedCliffordsUniformLossesSimulationStrategy(
            deepcopy(self.bs_permanent_calculator),
            self._experiment_configuration.uniform_transmissivity,
        )

    def _generate_gmf_general_losses_strategy(
        self,
    ) -> GeneralizedMeanFieldNonuniformLossesSimulationStrategy:
        approximated_modes_number = self._experiment_configuration.number_of_modes
        approximated_modes_number -= self._experiment_configuration.hierarchy_level

        return GeneralizedMeanFieldNonuniformLossesSimulationStrategy(
            bs_permanent_calculator=deepcopy(self.bs_permanent_calculator),
            approximated_modes_number=approximated_modes_number,
        )

    def _generate_gmf_uniform_losses_strategy(
        self,
    ) -> GeneralizedMeanFieldUniformLossesSimulationStrategy:
        return GeneralizedMeanFieldUniformLossesSimulationStrategy(
            bs_permanent_calculator=self._bs_permanent_calculator,
            uniform_transmissivity=self._experiment_configuration.uniform_transmissivity,
            hierarchy_level=self._experiment_configuration.hierarchy_level,
            threads_number=self._available_threads_number,
        )
