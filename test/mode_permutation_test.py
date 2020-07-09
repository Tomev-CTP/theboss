__author__ = 'Tomasz Rybotycki'

# TR TODO: Consider making this an actual test.

import numpy as np
import math
from scipy import special
from src.Boson_Sampling_Utilities import modes_state_to_particle_state, \
    particle_state_to_modes_state, calculate_permanent, generate_lossy_inputs
from src.BosonSamplingSimulator import BosonSamplingSimulator
from src.simulation_strategies.FixedLossSimulationStrategy import FixedLossSimulationStrategy

# Generate permutation matrix and define initial state.
permutation_matrix = np.zeros((5, 5))
initial_state = [1, 1, 1, 0, 0]

permutation_matrix[0][2] = 1
permutation_matrix[1][0] = 1
permutation_matrix[2][3] = 1
permutation_matrix[3][4] = 1
permutation_matrix[4][1] = 1

# Define some constants that will be used through the experiment.
NUMBER_OF_LOST_PARTICLES = 2
INITIAL_NUMBER_OF_PARTICLES = initial_state.count(1)
NUMBER_OF_PARTICLES_LEFT = INITIAL_NUMBER_OF_PARTICLES - NUMBER_OF_LOST_PARTICLES
NUMBER_OF_MODES = 5

# Define some variables for this scenario, to ease the computations.
possible_outcomes = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]


def main():
    exact_distribution = calculate_exact_distribution()
    print(f'Exact distribution: {exact_distribution}')

    approximate_distribution = calculate_approximate_distribution()
    print(f'Approximate distribution: {approximate_distribution}')


def calculate_exact_distribution() -> list:
    """
    This method will be used to calculate the exact distribution of lossy boson sampling experiment.
    The results will be returned as a table of probabilities of obtaining the outcome at i-th index.
    :return: List of probabilities of outcomes.
    """
    outcomes_probabilities = []
    for outcome in possible_outcomes:
        outcomes_probabilities.append(calculate_probability_of_outcome(outcome))
    return outcomes_probabilities


def calculate_probability_of_outcome(outcome: list) -> float:
    """
    Given input state and losses as described in the constants of this file calculate the probability
    of obtaining the outcome given as a parameter, after traversing lossy linear-optical channel.
    :param outcome: An outcome which probability of obtaining will be calculated.
    :return: Probability of obtaining given outcome in situation presented by by the
    """
    outcome_probability = 0  # Initialize with 0 for later debug purposes.
    outcome_state_in_particle_basis = modes_state_to_particle_state(outcome, NUMBER_OF_PARTICLES_LEFT)

    outcome_probability = calculate_probability_of_outcome_state_for_indistinguishable_photons(outcome_state_in_particle_basis)

    # Different states in particles-basis may give the same outcome state.
    outcome_probability *= math.factorial(NUMBER_OF_PARTICLES_LEFT)
    for i in range(NUMBER_OF_MODES):
        outcome_probability /= math.factorial(outcome[i])

    return outcome_probability


def calculate_probability_of_outcome_state_for_indistinguishable_photons(outcome_state_in_particle_basis: list) \
        -> float:
    copy_of_outcome_state = list(outcome_state_in_particle_basis[:])
    outcome_state_in_mode_basis = particle_state_to_modes_state(copy_of_outcome_state, NUMBER_OF_MODES)

    probability_of_outcome = 0

    # Symmetrize the input.
    lossy_inputs_list = generate_lossy_inputs(initial_state, NUMBER_OF_PARTICLES_LEFT)
    for lossy_input in lossy_inputs_list:
        subprobability = abs(calculate_permanent(
            calculate_submatrix_for_permanent_calculation(lossy_input, outcome_state_in_mode_basis))) ** 2
        for mode_occupation_number in lossy_input:
            subprobability /= math.factorial(mode_occupation_number)

        probability_of_outcome += subprobability

    # Normalization (read Brod & Oszmaniec 2019).
    probability_of_outcome /= math.factorial(NUMBER_OF_PARTICLES_LEFT)
    probability_of_outcome /= special.binom(INITIAL_NUMBER_OF_PARTICLES, NUMBER_OF_PARTICLES_LEFT)

    return probability_of_outcome


def calculate_submatrix_for_permanent_calculation(lossy_input: list, outcome_state_in_mode_basis: list) -> list:
    """
    In order to calculate exact distribution of a boson sampling experiment with interferometer denoted as U, a
    permanent of specific matrix has to be calculated. The matrix has no distinct name, and it's only description
    is a recipe how to construct it (described in e.g. Brod and Oszmaniec).
    :param lossy_input:
    :param outcome_state_in_mode_basis:
    :return: The submatrix for permanent calculator.
    """
    columns_permutation_submatrix = create_column_submatrix_of_interferometer_matrix(lossy_input)
    return create_rows_submatrix_of_interferometer_column_submatrix(outcome_state_in_mode_basis,
                                                                    columns_permutation_submatrix)


def create_column_submatrix_of_interferometer_matrix(lossy_input: list):
    columns_permutation_submatrix = 1j * np.zeros((NUMBER_OF_MODES, NUMBER_OF_PARTICLES_LEFT))

    # Copying occupation_number times the i-th column of permutation_matrix (or U in general).
    column_iterator = 0
    mode_number = 0

    for occupation_number in lossy_input:
        while occupation_number > 0:
            columns_permutation_submatrix[:, column_iterator] = permutation_matrix[:, mode_number]
            column_iterator += 1
            occupation_number -= 1
        mode_number += 1

    return columns_permutation_submatrix


def create_rows_submatrix_of_interferometer_column_submatrix(outcome_state_in_mode_basis: list,
                                                             columns_permutation_submatrix: list) -> list:
    permutation_submatrix = 1j * np.zeros((NUMBER_OF_PARTICLES_LEFT, NUMBER_OF_PARTICLES_LEFT))
    submatrix_row = 0

    # Copying occupation_number times the i-th row of columns_permutation_submatrix.
    for mode_number in range(NUMBER_OF_MODES):
        occupation_number = outcome_state_in_mode_basis[mode_number]
        while occupation_number > 0:
            permutation_submatrix[submatrix_row, :] = columns_permutation_submatrix[mode_number, :]
            occupation_number -= 1
            submatrix_row += 1

    return permutation_submatrix


def calculate_approximate_distribution(samples_number: int = 1000) -> list:
    """
    Prepares the approximate distribution using boson sampling simulation method described by
    Oszmaniec and Brod. Obviously higher number of samples will generate better approximation.
    :return: Approximate distribution as a list.
    """

    strategy = FixedLossSimulationStrategy(permutation_matrix, NUMBER_OF_PARTICLES_LEFT, NUMBER_OF_MODES)
    simulator = BosonSamplingSimulator(NUMBER_OF_PARTICLES_LEFT, INITIAL_NUMBER_OF_PARTICLES, NUMBER_OF_MODES, strategy)
    outcomes_probabilities = [0, 0, 0]
    for i in range(samples_number):
        result = simulator.get_classical_simulation_results()

        for j in range(len(possible_outcomes)):
            # Check if obtained result is one of possible outcomes.
            if not (result == possible_outcomes[j]).__contains__(False):
                outcomes_probabilities[j] += 1
                break

    for i in range(len(outcomes_probabilities)):
        outcomes_probabilities[i] /= samples_number

    return outcomes_probabilities


if __name__ == '__main__':
    main()
