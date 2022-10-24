__author__ = "Tomasz Rybotycki"

from typing import List, Iterable, Sequence

from numpy import zeros, asarray

from theboss.boson_sampling_simulator import BosonSamplingSimulator
from theboss.boson_sampling_utilities import generate_possible_states
from theboss.distribution_calculators.bs_distribution_calculator_interface import (
    BosonSamplingExperimentConfiguration,
    BSDistributionCalculatorInterface,
)
from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)


class BSSampleBasedDistributionCalculator(BSDistributionCalculatorInterface):
    """
    A class for computing the frequencies of the specific outcomes of BS experiment.
    It's called DistributionCalculator only to be possibly used with the same interface.
    """

    def __init__(
        self,
        experiment_configuration: BosonSamplingExperimentConfiguration,
        strategy: SimulationStrategyInterface,
        samples_number: int = 5000,
        outcomes: List[Sequence[int]] = None,
    ) -> None:
        self._configuration: BosonSamplingExperimentConfiguration = (
            experiment_configuration
        )
        self._strategy: SimulationStrategyInterface = strategy
        self._outcomes: List[Sequence[int]] = outcomes
        self._samples_number: int = samples_number

    @property
    def configuration(self) -> BosonSamplingExperimentConfiguration:
        """
        Configuration of the BS experiment.
        """
        return self._configuration

    @configuration.setter
    def configuration(
        self, new_configuration: BosonSamplingExperimentConfiguration
    ) -> None:
        self._configuration = new_configuration

    @property
    def strategy(self) -> SimulationStrategyInterface:
        """
        Strategy used for the classical BS simulation.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: SimulationStrategyInterface) -> None:
        self._strategy = new_strategy

    @property
    def outcomes(self) -> List[Sequence[int]]:
        """
        The outcomes of the BS experiment that will be taken into account during the
        frequencies' computation.
        """
        return self._outcomes

    @outcomes.setter
    def outcomes(self, new_outcomes: List[Sequence[int]]) -> None:
        self._outcomes = new_outcomes

    @property
    def samples_number(self) -> int:
        """
        The number of samples used in frequencies computation.
        """
        return self._samples_number

    @samples_number.setter
    def samples_number(self, new_samples_number: int) -> None:
        self._samples_number = new_samples_number

    def calculate_distribution(self) -> List[float]:
        return self.calculate_approximate_distribution()

    def calculate_probabilities_of_outcomes(
        self, outcomes: Iterable[Iterable[int]]
    ) -> List[float]:
        self._outcomes = outcomes
        return self.calculate_approximate_distribution()

    def calculate_approximate_distribution(
        self, samples_number: int = 5000
    ) -> List[float]:
        """
        Prepares the approximate distribution using boson sampling simulation method.
        Obviously higher number of samples will generate better approximation.

        .. note::
            Approximate distribution computes probabilities of the ``outcomes`` if
            this variable is set. In order to compute whole distribution, all possible
            outcomes has to be set in the ``outcomes`` variable.

        :return:
            Approximate distribution as a list of probabilities.
        """

        if self._outcomes is not None:
            possible_outcomes = self._outcomes
        else:
            possible_outcomes = generate_possible_states(
                self.configuration.number_of_particles_left,
                self.configuration.number_of_modes,
            )

        simulator = BosonSamplingSimulator(self._strategy)

        outcomes_probabilities = zeros(len(possible_outcomes), dtype=float)

        samples = simulator.get_classical_simulation_results(
            self.configuration.initial_state, samples_number
        )
        for sample in samples:
            for j in range(len(possible_outcomes)):
                # Check if obtained result is one of possible outcomes.
                if all(
                    asarray(sample) == possible_outcomes[j]
                ):  # Expect all elements of resultant list to be True.
                    outcomes_probabilities[j] += 1
                    break

        outcomes_probabilities = outcomes_probabilities / samples_number

        return list(outcomes_probabilities)

    def get_outcomes_in_proper_order(self) -> List[Sequence[int]]:
        return self._outcomes
