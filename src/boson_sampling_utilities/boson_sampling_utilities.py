__author__ = "Tomasz Rybotycki"

# TR TODO: Consider making this file a package along with exact distribution calculator.

import itertools
from typing import List, Optional

from numpy import array, asarray, block, complex128, diag, eye, int64, ndarray, power, \
    sqrt, transpose, zeros, zeros_like, square, flip, pi, ones, exp
from numpy.linalg import svd
from scipy.special import binom
from numpy.random import rand

from ..quantum_computations_utilities import compute_qft_matrix


def particle_state_to_modes_state(particle_state: ndarray,
                                  observed_modes_number: int) -> ndarray:
    modes_state = zeros(observed_modes_number)

    # Adding the particle to it's mode.
    for particles_mode in asarray(particle_state, dtype=int64):
        modes_state[particles_mode] += 1

    return modes_state


def modes_state_to_particle_state(mode_state: ndarray,
                                  particles_number: int) -> ndarray:
    """
        Return given mode-basis state in particle basis.

        :param mode_state: Input state in mode-basis.
        :param particles_number: Number of particles.
        :return: Given mode-basis state in particle basis.
    """

    number_of_observed_modes = len(mode_state)
    modes = mode_state.copy()
    particles_state = zeros(particles_number)

    i = k = 0
    while i < number_of_observed_modes:

        if modes[i] > 0:
            modes[i] -= 1
            particles_state[k] = i
            k += 1
        else:
            i += 1

    return particles_state


def generate_possible_outputs(number_of_particles: int, number_of_modes: int,
                              consider_loses: bool = False) \
        -> List[ndarray]:
    if number_of_particles < 0 or number_of_modes < 1:
        return []
    if number_of_particles == 0:
        return [zeros(number_of_modes)]

    outputs = []
    starting_number_of_particles = number_of_particles

    if consider_loses:
        starting_number_of_particles = 0

    for n in range(starting_number_of_particles, number_of_particles + 1):
        outputs.extend(generate_possible_n_particle_outputs(n, number_of_modes))

    return outputs


def generate_possible_n_particle_outputs(number_of_particles: int,
                                         number_of_modes: int) -> List[ndarray]:
    outputs = []

    output = zeros(number_of_modes)
    output[0] = number_of_particles
    outputs.append(output)

    while outputs[-1][number_of_modes - 1] < number_of_particles:

        k = number_of_modes - 1
        while outputs[-1][k - 1] == 0:
            k -= 1

        output = outputs[-1].copy()
        output[k - 1] -= 1
        output[k:] = 0
        output[k] = number_of_particles - sum(output)

        outputs.append(output)

    sorted_outputs = sorted([tuple(output) for output in outputs])

    return [array(output) for output in sorted_outputs]


def generate_lossy_inputs(initial_state: ndarray, number_of_particles_left: int) -> \
List[ndarray]:
    """
        From initial state generate all possible input states after losses application.
        :param initial_state: The state we start with.
        :param number_of_particles_left: Number of particles after losses application.
        :return: A list of lists representing initial states after losses.
    """
    x0 = []
    number_of_modes = len(initial_state)
    initial_number_of_particles = sum(initial_state)
    for i in range(number_of_modes):
        x0.extend([i] * int(initial_state[i]))

    lossy_inputs_list = []
    lossy_inputs_hashes = []

    if sum(initial_state) == 0:
        return [initial_state]

    # Symmetrization.
    for combination in itertools.combinations(list(range(initial_number_of_particles)),
                                              number_of_particles_left):
        lossy_input_in_particle_basis = array([x0[el] for el in combination],
                                              dtype=int64)

        lossy_input = particle_state_to_modes_state(lossy_input_in_particle_basis,
                                                    number_of_modes)

        # Check if calculated lossy input is already in the list. If not, add it.
        lossy_input_hash = hash(tuple(lossy_input))
        if lossy_input_hash not in lossy_inputs_hashes:
            lossy_inputs_list.append(lossy_input)
            lossy_inputs_hashes.append(lossy_input_hash)

    return lossy_inputs_list


def calculate_number_of_possible_n_particle_m_mode_output_states(n: int, m: int) -> int:
    """
        Calculates the number of possible output states with n particles placed around m modes.

        This is basically the same answer as to in how many possible combinations can we put n objects in m bins. It's
        also a dimension of n-particle m-mode bosonic space. Stars-and-bars argument applies here.

        :param n: Number of particles.
        :param m: Number of modes.
        :return: Dimension of n-particle m-mode bosonic space.
    """
    return round(binom(n + m - 1, n))


def calculate_number_of_possible_lossy_n_particle_m_mode_output_states(n: int,
                                                                       m: int) -> int:
    """
        Calculates the number of possible output states with N <= n particles placed around m modes.

        :param n: Number of particles.
        :param m: Number of modes.
        :return: Dimension of n-particle m-mode bosonic space.
    """
    states_number = 0
    for N in range(n + 1):
        states_number += binom(N + m - 1, N)
    return states_number


def get_modes_transmissivity_values_from_matrix(lossy_interferometer_matrix: ndarray) -> \
List[float]:
    v_matrix, singular_values, u_matrix = svd(lossy_interferometer_matrix)
    return square(flip(singular_values))


def _calculate_singular_values_matrix_expansion(
        singular_values_vector: ndarray) -> ndarray:
    vector_of_squared_expansions = 1.0 - power(singular_values_vector, 2)
    for i in range(len(vector_of_squared_expansions)):
        if vector_of_squared_expansions[i] < 0:
            vector_of_squared_expansions[i] = 0

    expansion_values = sqrt(vector_of_squared_expansions)

    return diag(expansion_values)


def prepare_interferometer_matrix_in_expanded_space(
        interferometer_matrix: ndarray) -> ndarray:
    v_matrix, singular_values, u_matrix = svd(interferometer_matrix)

    expansions_zeros = zeros_like(v_matrix)
    expansions_ones = eye(len(v_matrix))
    expanded_v = block(
        [[v_matrix, expansions_zeros], [expansions_zeros, expansions_ones]])
    expanded_u = block(
        [[u_matrix, expansions_zeros], [expansions_zeros, expansions_ones]])
    singular_values_matrix_expansion = _calculate_singular_values_matrix_expansion(
        singular_values)
    singular_values_expanded_matrix = block(
        [[diag(singular_values), singular_values_matrix_expansion],
         [singular_values_matrix_expansion, diag(singular_values)]])
    return expanded_v @ singular_values_expanded_matrix @ expanded_u


def compute_state_types(modes_number: int, particles_number: int,
                        losses: bool = False) -> List[List[int]]:
    # Partitions generating code.
    # Taken from https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning/10036764
    def partitions(n, I=1):
        yield (n,)
        for i in range(I, n // 2 + 1):
            for p in partitions(n - i, i):
                yield (i,) + p

    all_partitions = list(partitions(particles_number))

    if losses:
        for i in range(particles_number):
            all_partitions += list(partitions(i))

    state_types = []

    for partition in all_partitions:
        if len(partition) > modes_number:
            continue
        # We describe state type by a vector in descending order.
        state_type = sorted(partition, reverse=True)
        state_types.append(state_type)

    return state_types


def compute_number_of_state_types(modes_number: int,
                                  particles_number: int,
                                  losses = False) -> int:
    state_types_number = 0

    for k in range(1, modes_number + 1):
        state_types_number += \
            compute_number_of_k_element_integer_n_partitions(k, particles_number)

    if not losses:
        return state_types_number

    for particles_num in range(particles_number):
        for k in range(1, modes_number + 1):
            state_types_number += \
                compute_number_of_k_element_integer_n_partitions(k, particles_num)

    return state_types_number

def compute_number_of_k_element_integer_n_partitions(k: int, n: int) -> int:

    if k == 1:
        return 1

    if k > n or n == 0 or k < 1:
        return 0

    integer_partitions_number = \
        compute_number_of_k_element_integer_n_partitions(k, n - k)
    integer_partitions_number += \
        compute_number_of_k_element_integer_n_partitions(k - 1, n - 1)

    return integer_partitions_number


def compute_maximally_unbalanced_types(modes_number: int, particles_number: int) -> \
List[List[int]]:
    maximally_unbalanced_types = []
    all_types = compute_state_types(particles_number=particles_number,
                                    modes_number=modes_number)

    for state_type in all_types:
        if state_type.count(1) == len(state_type) - 1 or state_type.count(1) == len(
                state_type):
            maximally_unbalanced_types.append(state_type)

    return maximally_unbalanced_types


def generate_qft_matrix_for_first_m_modes(m: int, all_modes_number: int) -> ndarray:
    small_qft_matrix = compute_qft_matrix(m)
    qft_matrix = eye(all_modes_number, dtype=complex128)
    qft_matrix[0:m, 0:m] = small_qft_matrix
    return qft_matrix


def generate_random_phases_matrix_for_first_m_modes(m: int, all_modes_number: int) \
        -> ndarray:
    random_phases = ones(all_modes_number, dtype=complex128)  # [1, 1, 1, 1, 1, 1]
    random_phases[0:m] = exp(1j * 2 * pi * rand(m))
    return diag(random_phases)

class EffectiveScatteringMatrixCalculator:
    """
        In many methods of Boson Sampling simulations an effective scattering matrix has to be calculated. Therefore
        I decided to implement an calculator that'd be used if every single one of these methods.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        if output_state is None:
            output_state = array([], dtype=int64)
        if input_state is None:
            input_state = array([], dtype=int64)
        self.__matrix = matrix
        self.__input_state = input_state
        self.__output_state = output_state

    @property
    def matrix(self) -> ndarray:
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix: ndarray) -> None:
        self.__matrix = matrix

    @property
    def input_state(self) -> ndarray:
        return self.__input_state

    @input_state.setter
    def input_state(self, input_state: ndarray) -> None:
        self.__input_state = asarray(input_state, dtype=int64)

    @property
    def output_state(self) -> ndarray:
        return self.__output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self.__output_state = asarray(output_state, dtype=int64)

    def calculate(self) -> ndarray:
        transposed_input_matrix = transpose(self.__matrix)
        helper_mtx = []

        for index_of_column_to_insert in range(len(self.__input_state)):
            helper_mtx += [transposed_input_matrix[index_of_column_to_insert]] * \
                          int(self.__input_state[index_of_column_to_insert])

        helper_mtx = transpose(array(helper_mtx, dtype=complex128))

        effective_scattering_matrix = []

        for index_of_row_to_insert in range(len(self.__output_state)):
            effective_scattering_matrix += [helper_mtx[index_of_row_to_insert]] * \
                                           int(self.__output_state[
                                                   index_of_row_to_insert])

        return array(effective_scattering_matrix, dtype=complex128)
