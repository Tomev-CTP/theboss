__author__ = "Tomasz Rybotycki"

from typing import List, Iterable

from numpy import float64, ndarray, zeros

from ..boson_sampling_simulator import BosonSamplingSimulator
from ..boson_sampling_utilities.boson_sampling_utilities import generate_possible_outputs
from ..distribution_calculators.bs_distribution_calculator_interface import \
    BosonSamplingExperimentConfiguration, BSDistributionCalculatorInterface
from ..simulation_strategies.simulation_strategy_interface import SimulationStrategyInterface


class BSSampleBasedDistributionCalculator(BSDistributionCalculatorInterface):
    def __init__(self, experiment_configuration: BosonSamplingExperimentConfiguration,
                 strategy: SimulationStrategyInterface, samples_number=5000, outcomes=None) -> None:
        self._configuration = experiment_configuration
        self._strategy = strategy
        self._outcomes = outcomes
        self._samples_number = samples_number

    @property
    def configuration(self) -> BosonSamplingExperimentConfiguration:
        return self._configuration

    @configuration.setter
    def configuration(self, new_configuration: BosonSamplingExperimentConfiguration) -> None:
        self._configuration = new_configuration

    @property
    def strategy(self) -> SimulationStrategyInterface:
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: SimulationStrategyInterface) -> None:
        self._strategy = new_strategy

    @property
    def outcomes(self) -> List[ndarray]:
        return self._outcomes

    @outcomes.setter
    def outcomes(self, new_outcomes: List[ndarray]) -> None:
        self._outcomes = new_outcomes

    @property
    def samples_number(self) -> int:
        return self._samples_number

    @samples_number.setter
    def samples_number(self, new_samples_number: int) -> None:
        self._samples_number = new_samples_number

    def calculate_distribution(self) -> List[float]:
        return self.calculate_approximate_distribution()

    def calculate_probabilities_of_outcomes(self, outcomes: Iterable[Iterable[int]]) -> List[float]:
        self._outcomes = outcomes
        return self.calculate_approximate_distribution()

    def calculate_approximate_distribution(self, samples_number: int = 5000) -> List[float]:
        """
            Prepares the approximate distribution using boson sampling simulation method described by
            Oszmaniec and Brod. Obviously higher number of samples will generate better approximation.
            :return: Approximate distribution as a list.
        """

        if self._outcomes is not None:
            possible_outcomes = self._outcomes
        else:
            possible_outcomes = generate_possible_outputs(self.configuration.number_of_particles_left,
                                                          self.configuration.number_of_modes)

        simulator = BosonSamplingSimulator(self._strategy)

        outcomes_probabilities = zeros(len(possible_outcomes), dtype=float64)

        samples = simulator.get_classical_simulation_results(self.configuration.initial_state, samples_number)
        for sample in samples:
            for j in range(len(possible_outcomes)):
                # Check if obtained result is one of possible outcomes.
                if all(sample == possible_outcomes[j]):  # Expect all elements of resultant list to be True.
                    outcomes_probabilities[j] += 1
                    break

        outcomes_probabilities = outcomes_probabilities / samples_number

        return list(outcomes_probabilities)

    def get_outcomes_in_proper_order(self) -> List[ndarray]:
        return self._outcomes
