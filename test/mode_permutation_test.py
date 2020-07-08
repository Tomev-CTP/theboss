__author__ = 'Tomasz Rybotycki'

import numpy as np
import math
import itertools
from scipy import special
from src.Boson_Sampling_Utilities import modes_state_to_particle_state, \
    particle_state_to_modes_state, calculate_permanent, generate_possible_outputs
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
#possible_outcomes = [[1, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 1, 0, 0, 1]]
possible_outcomes = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]]


def main():
    exact_distribution = calculate_exact_distribution()
    print(f'Exact distribution: {exact_distribution}')
    strategy = FixedLossSimulationStrategy(permutation_matrix, NUMBER_OF_PARTICLES_LEFT, NUMBER_OF_MODES)
    simulator = BosonSamplingSimulator(NUMBER_OF_PARTICLES_LEFT, INITIAL_NUMBER_OF_PARTICLES,\
                                       NUMBER_OF_MODES, strategy)

    outcomes_probabilities = [0, 0, 0]
    samples_number = 1000000
    for i in range(samples_number):
        result = simulator.get_classical_simulation_results()

        for j in range(len(possible_outcomes)):
            if not (result == possible_outcomes[j]).__contains__(False): # Check if obtained result is one of possible outcomes
                outcomes_probabilities[j] += 1
                break

    for i in range(len(outcomes_probabilities)):
        outcomes_probabilities[i] /= samples_number

    print(f'Approximate distribution: {outcomes_probabilities}')


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
    outcome_probability = 0
    outcome_state_in_particle_basis = modes_state_to_particle_state(outcome, NUMBER_OF_PARTICLES_LEFT)

    outcome_probability = calculate_probability_of_outcome_state_for_indistinguishable_photons(outcome_state_in_particle_basis)

    # Different states in particles-basis may give the same outcome state.
    outcome_probability *= math.factorial(NUMBER_OF_PARTICLES_LEFT)
    for i in range(NUMBER_OF_MODES):
        outcome_probability /= math.factorial(outcome[i])

    return outcome_probability


def calculate_probability_of_outcome_state_for_indistinguishable_photons(outcome_state_in_particle_basis: list) \
        -> float:
    # numbers of modes from zero

    r_new = list(outcome_state_in_particle_basis[:])  # to create a new list
    T_No = particle_state_to_modes_state(r_new, NUMBER_OF_MODES)  # proposed sample in modes basis

    prob = 0.
    # symmetrization of the input
    lossy_inputs_list = generate_lossy_inputs()
    for lossy_input in lossy_inputs_list:
        # probability in modes-basis (read Brod & Oszmaniec 2019)
        subprob = abs(calculate_permanent(calculate_submatrix_for_permanent_calculation(lossy_input, T_No))) ** 2
        for S_j in lossy_input:
            subprob /= math.factorial(S_j)

        prob += subprob

    # normalization (read Brod & Oszmaniec 2019)
    prob /= math.factorial(NUMBER_OF_PARTICLES_LEFT)
    prob /= special.binom(INITIAL_NUMBER_OF_PARTICLES, NUMBER_OF_PARTICLES_LEFT)

    return prob


# All the possible states from initial_state with NUMBER_OF_PARTICLES_LEFT photons
def generate_lossy_inputs():
    x0 = []
    for i in range(NUMBER_OF_MODES):
        for j in range(int(initial_state[i])):
            x0.append(i)  # modes numbered from 0

    lossy_inputs_table = []

    # symmetrization
    for obj in itertools.combinations(list(range(INITIAL_NUMBER_OF_PARTICLES)), NUMBER_OF_PARTICLES_LEFT):
        x = []  # all possible entries with No of photons from S
        for el in obj:
            x.append(x0[el])

        lossy_input = particle_state_to_modes_state(x, NUMBER_OF_MODES)
        if all(list(lossy_inputs_table[el]) != list(lossy_input) for el in range(len(lossy_inputs_table))):
            lossy_inputs_table.append(lossy_input)

    return lossy_inputs_table


def calculate_submatrix_for_permanent_calculation(lossy_input, r):
    U_S = 1j * np.zeros((NUMBER_OF_MODES, NUMBER_OF_PARTICLES_LEFT))
    column = 0

    # copying s_i times the i-th column of U
    for i in range(NUMBER_OF_MODES):

        s_i = lossy_input[i]
        while s_i > 0:
            U_S[:, column] = permutation_matrix[:, i]
            column += 1
            s_i -= 1

    U_Sr = 1j * np.zeros((NUMBER_OF_PARTICLES_LEFT, NUMBER_OF_PARTICLES_LEFT))
    row = 0

    # copying r_i times the i-th row of U_S
    for i in range(NUMBER_OF_MODES):

        r_i = r[i]
        while r_i > 0:
            U_Sr[row, :] = U_S[i, :]
            row += 1
            r_i -= 1

    return U_Sr


# all the inputs with 'l' particles on 'n' modes
def generate_n_mode_inputs(m, n, l):
    # n has to be lower than m !!!

    inputs = []

    n_input = np.zeros(n)
    n_input[0] = l
    m_input = np.zeros(m)
    m_input[0] = l
    inputs.append(m_input)

    # a loop generating new possible inputs
    while (n_input[n - 1] != l):

        k = n - 1
        while (n_input[k - 1] == 0):
            k -= 1

        n_input[k - 1] -= 1
        n_input[k:] = 0
        n_input[k] = l - sum(n_input)

        m_input = np.zeros(m)
        m_input[:n] = list(n_input[:])

        inputs.append(m_input)

    return inputs


if __name__ == '__main__':
    main()
