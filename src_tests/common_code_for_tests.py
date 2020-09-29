__author__ = 'Tomasz Rybotycki'

from numpy import zeros
from typing import List

from src.LossyBosonSamplingExactDistributionCalculators import BosonSamplingExperimentConfiguration
from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.simulation_strategies.SimulationStrategy import SimulationStrategy
from src.Boson_Sampling_Utilities import generate_possible_outputs


class ApproximateDistributionCalculator:
    def __init__(self, experiment_configuration: BosonSamplingExperimentConfiguration,
                 strategy: SimulationStrategy):
        self.configuration = experiment_configuration
        self.strategy = strategy

    def calculate_approximate_distribution(self, samples_number: int = 5000) -> List[float]:
        """
            Prepares the approximate distribution using boson sampling simulation method described by
            Oszmaniec and Brod. Obviously higher number of samples will generate better approximation.
            :return: Approximate distribution as a list.
        """
        possible_outcomes = generate_possible_outputs(self.configuration.number_of_particles_left,
                                                      self.configuration.number_of_modes)

        simulator = BosonSamplingSimulator(self.configuration.number_of_particles_left,
                                           self.configuration.initial_number_of_particles,
                                           self.configuration.number_of_modes, self.strategy)

        outcomes_probabilities = zeros(len(possible_outcomes))

        for i in range(samples_number):


            print(f'Sample number {i} of {samples_number}.')

            result = simulator.get_classical_simulation_results(self.configuration.initial_state)

            for j in range(len(possible_outcomes)):
                # Check if obtained result is one of possible outcomes.
                if all(result == possible_outcomes[j]):  # Expect all elements of resultant list to be True.
                    outcomes_probabilities[j] += 1
                    break

        for i in range(len(outcomes_probabilities)):
            outcomes_probabilities[i] /= samples_number

        return outcomes_probabilities

