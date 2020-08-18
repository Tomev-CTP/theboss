__author__ = 'Tomasz Rybotycki'

# TR TODO: This could be a part ob Boson_Sampling_Utilities package.

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List

from numpy import ndarray, zeros
from scipy import special

from src.Boson_Sampling_Utilities import calculate_permanent, generate_lossy_inputs, generate_possible_outputs, \
    modes_state_to_particle_state, particle_state_to_modes_state


@dataclass
class BosonSamplingExperimentConfiguration:
    interferometer_matrix: ndarray  # A matrix describing interferometer.
    initial_state: ndarray
    initial_number_of_particles: int
    number_of_modes: int
    number_of_particles_lost: int
    number_of_particles_left: int
    probability_of_uniform_loss: float = 0


class BosonSamplingWithFixedLossesExactDistributionCalculator:
    def __init__(self, configuration: BosonSamplingExperimentConfiguration) -> None:
        self.configuration = deepcopy(configuration)

    def get_outcomes_in_proper_order(self) -> List[List[int]]:
        return generate_possible_outputs(self.configuration.number_of_particles_left,
                                         self.configuration.number_of_modes)

    def calculate_exact_distribution(self) -> List[List[float]]:
        """
        This method will be used to calculate the exact distribution of lossy boson sampling experiment.
        The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
        :return: List of probabilities of outcomes.
        """
        possible_outcomes = generate_possible_outputs(self.configuration.number_of_particles_left,
                                                      self.configuration.number_of_modes)

        outcomes_probabilities = []
        for outcome in possible_outcomes:
            outcomes_probabilities.append(self.__calculate_probability_of_outcome(outcome))
        return outcomes_probabilities

    def __calculate_probability_of_outcome(self, outcome: ndarray) -> float:
        """
        Given input state and losses as described in the constants of this file calculate the probability
        of obtaining the outcome given as a parameter, after traversing lossy linear-optical channel.
        :param outcome: An outcome which probability of obtaining will be calculated.
        :return: Probability of obtaining given outcome in situation presented by by the
        """
        outcome_probability = 0  # Initialize with 0 for later debug purposes.
        outcome_state_in_particle_basis = modes_state_to_particle_state(outcome,
                                                                        self.configuration.number_of_particles_left)

        outcome_probability = self.__calculate_probability_of_outcome_state_for_indistinguishable_photons(
            outcome_state_in_particle_basis)

        # Different states in particles-basis may give the same outcome state.
        outcome_probability *= math.factorial(self.configuration.number_of_particles_left)
        for i in range(self.configuration.number_of_modes):
            outcome_probability /= math.factorial(outcome[i])

        return outcome_probability

    def __calculate_probability_of_outcome_state_for_indistinguishable_photons(
            self, outcome_state_in_particle_basis: ndarray) -> float:
        copy_of_outcome_state = outcome_state_in_particle_basis.copy()
        outcome_state_in_mode_basis = particle_state_to_modes_state(copy_of_outcome_state,
                                                                    self.configuration.number_of_modes)
        probability_of_outcome = 0

        # Symmetrize the input.
        lossy_inputs_list = generate_lossy_inputs(self.configuration.initial_state,
                                                  self.configuration.number_of_particles_left)
        for lossy_input in lossy_inputs_list:
            subprobability = abs(calculate_permanent(
                self.__count_effective_boson_scattering_matrix(lossy_input, outcome_state_in_mode_basis))) ** 2
            for mode_occupation_number in lossy_input:
                subprobability /= math.factorial(mode_occupation_number)

            probability_of_outcome += subprobability

        # Normalization (read Brod & Oszmaniec 2019).
        probability_of_outcome /= math.factorial(self.configuration.number_of_particles_left)
        probability_of_outcome /= special.binom(self.configuration.initial_number_of_particles,
                                                self.configuration.number_of_particles_left)

        return probability_of_outcome

    def __count_effective_boson_scattering_matrix(
            self, lossy_input: ndarray, outcome_state_in_mode_basis: ndarray) -> ndarray:
        """
        In order to calculate exact distribution of a boson sampling experiment with interferometer denoted as U, a
        permanent of specific matrix has to be calculated. The matrix has no distinct name, and it's only description
        is a recipe how to construct it (described in e.g. Brod and Oszmaniec).
        :param lossy_input:
        :param outcome_state_in_mode_basis:
        :return: The submatrix for permanent calculator.
        """
        columns_permutation_submatrix = self.__create_column_submatrix_of_effective_boson_scattering_matrix(lossy_input)
        return self.__create_rows_submatrix_of_effective_boson_scattering_matrix(outcome_state_in_mode_basis,
                                                                                 columns_permutation_submatrix)

    def __create_column_submatrix_of_effective_boson_scattering_matrix(self, lossy_input: ndarray) -> ndarray:
        columns_permutation_submatrix = 1j * zeros((self.configuration.number_of_modes,
                                                    self.configuration.number_of_particles_left))

        # Copying occupation_number times the i-th column of permutation_matrix (or U in general).
        column_iterator = 0
        mode_number = 0

        for occupation_number in lossy_input:
            while occupation_number > 0:
                columns_permutation_submatrix[:, column_iterator] = \
                    self.configuration.interferometer_matrix[:, mode_number]
                column_iterator += 1
                occupation_number -= 1
            mode_number += 1

        return columns_permutation_submatrix

    def __create_rows_submatrix_of_effective_boson_scattering_matrix(
            self, outcome_state_in_mode_basis: ndarray, columns_permutation_submatrix: ndarray) -> ndarray:
        permutation_submatrix = 1j * zeros((self.configuration.number_of_particles_left,
                                            self.configuration.number_of_particles_left))
        submatrix_row = 0

        # Copying occupation_number times the i-th row of columns_permutation_submatrix.
        for mode_number in range(self.configuration.number_of_modes):
            occupation_number = outcome_state_in_mode_basis[mode_number]
            while occupation_number > 0:
                permutation_submatrix[submatrix_row, :] = columns_permutation_submatrix[mode_number, :]
                occupation_number -= 1
                submatrix_row += 1

        return permutation_submatrix


class BosonSamplingWithUniformLossesExactDistributionCalculator \
            (BosonSamplingWithFixedLossesExactDistributionCalculator):
    def __init__(self, configuration: BosonSamplingExperimentConfiguration) -> None:
        self.configuration = deepcopy(configuration)

    def calculate_exact_distribution(self) -> List[List[float]]:
        """
        This method will be used to calculate the exact distribution of lossy boson sampling experiment.
        The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
        :return: List of probabilities of outcomes.
        """
        possible_outcomes = []
        exact_distribution = []

        # Using eta, n and l notation from the paper for readability purposes.
        n = self.configuration.initial_number_of_particles
        eta = self.configuration.probability_of_uniform_loss
        for number_of_particles_left in range(n + 1):  # +1 to include situation with all particles left.

            l = number_of_particles_left

            subconfiguration = deepcopy(self.configuration)

            subconfiguration.number_of_particles_left = number_of_particles_left
            subconfiguration.number_of_particles_lost = n - l
            subdistribution_calculator = \
                BosonSamplingWithFixedLossesExactDistributionCalculator(subconfiguration)
            possible_outcomes.extend(subdistribution_calculator.get_outcomes_in_proper_order())
            subdistribution = subdistribution_calculator.calculate_exact_distribution()
            subdistribution_weight = pow(eta, l) * special.binom(n, l) * pow(1.0 - eta, n - l)
            subdistribution = [el * subdistribution_weight for el in subdistribution]

            exact_distribution.extend(subdistribution)

        return exact_distribution

    def get_outcomes_in_proper_order(self) -> List[List[int]]:
        possible_outcomes = []

        for number_of_particles_left in range(self.configuration.initial_number_of_particles + 1):
            subconfiguration = deepcopy(self.configuration)
            subconfiguration.number_of_particles_left = number_of_particles_left
            subdistribution_calculator = \
                BosonSamplingWithFixedLossesExactDistributionCalculator(subconfiguration)
            possible_outcomes.extend(subdistribution_calculator.get_outcomes_in_proper_order())

        return possible_outcomes
