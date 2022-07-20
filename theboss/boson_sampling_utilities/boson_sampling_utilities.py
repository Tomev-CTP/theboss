__author__ = "Tomasz Rybotycki"

"""
    This script contains various auxiliary methods useful for boson sampling
    experiments.
"""

import itertools
from typing import List, Optional, Sequence, Tuple, Set

from numpy import (
    ndarray,
    array,
    block,
    complex128,
    diag,
    eye,
    power,
    sqrt,
    transpose,
    zeros_like,
    square,
    flip,
    pi,
    ones,
    exp,
)
from numpy.linalg import svd
from scipy.special import binom, factorial
from numpy.random import rand

from theboss.quantum_computations_utilities import compute_qft_matrix


def mode_assignment_to_mode_occupation(
    modes_assignment: Sequence[int], observed_modes_number: int = 0
) -> Tuple[int]:
    """
    Given a bosonic "state" in a mode assignment representation, return the state in the
    mode occupation description.

    :param modes_assignment:
        A "state" in a mode assignment representation.

    :param observed_modes_number:
        Number of observed modes. Necessary if it's greater than suggested by given
        state.

    :return:
        The state in a mode occupation representation (as a tuple).
    """
    if observed_modes_number == 0:
        observed_modes_number = max(modes_assignment)

    modes_occupation = [0 for _ in range(observed_modes_number)]

    for particles_mode in modes_assignment:
        modes_occupation[particles_mode] += 1

    return tuple(modes_occupation)


def mode_occupation_to_mode_assignment(mode_occupation: Sequence[int]) -> Tuple[int]:
    """
        Return the state (given in mode occupation representation) in the mode
        assignment representation.

        :param mode_occupation:
            Input state in mode-basis.

        :return:
            Given mode-basis state in particle basis (as a tuple).
    """
    mode_assignment = tuple()

    for i in range(len(mode_occupation)):
        for j in range(mode_occupation[i]):
            mode_assignment += i

    return mode_assignment


def generate_possible_states(
    particles_number: int, modes_number: int, losses: bool = False
) -> List[Tuple[int]]:
    """
    This method generates all possible :math:`m`-mode states. By default, it's
    restricted to only :math:`n`-particle states, but it can also return lossy states.
    
    :param particles_number:
        The maximal number :math:`n` of particles.
    :param modes_number:
        The number :math:`m` of considered modes.
    :param losses:
        A flag for lossy states generation.

    :return:
        A list of possible (lossy) states (as tuples).
    """
    if particles_number < 0 or modes_number < 1:
        return []
    if particles_number == 0:
        return [tuple([0 for _ in range(modes_number)])]

    states = []
    starting_number_of_particles = particles_number

    if losses:
        starting_number_of_particles = 0

    for particles_number in range(starting_number_of_particles, particles_number + 1):
        states.extend(_generate_possible_n_particle_states(particles_number, modes_number))

    return states


def _generate_possible_n_particle_states(
    n: int, modes_number: int
) -> List[Tuple[int]]:
    """
    Generates all possible :math:`n` particle states.


    :param n:
    :param modes_number:
    :return:
    """
    states = []

    state = [0 for _ in range(modes_number)]
    state[0] = n
    states.append(state)

    while states[-1][modes_number - 1] < n:

        k = modes_number - 1
        while states[-1][k - 1] == 0:
            k -= 1

        state = states[-1].copy()
        state[k - 1] -= 1
        state[k:] = 0
        state[k] = n - sum(state)

        states.append(state)

    sorted_states = sorted([tuple(output) for output in states], reverse=True)

    return sorted_states


def generate_lossy_inputs(
    initial_state: Sequence[int], number_of_particles_left: int
) -> List[List[int]]:
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
        return [list(initial_state)]

    # Symmetrization.
    for combination in itertools.combinations(
        list(range(initial_number_of_particles)), number_of_particles_left
    ):
        lossy_input_in_particle_basis = array(
            [x0[el] for el in combination], dtype=int
        )

        lossy_input = mode_assignment_to_mode_occupation(
            lossy_input_in_particle_basis, number_of_modes
        )

        # Check if calculated lossy input is already in the list. If not, add it.
        lossy_input_hash = hash(tuple(lossy_input))
        if lossy_input_hash not in lossy_inputs_hashes:
            lossy_inputs_list.append(lossy_input)
            lossy_inputs_hashes.append(lossy_input_hash)

    return lossy_inputs_list


def bosonic_space_dimension(n: int, m: int) -> int:
    """
        Calculates the number of possible output states with n particles placed around m
        modes.

        This is basically the same answer as to in how many possible combinations can we
        put n objects in m bins. It's also a dimension of n-particle m-mode bosonic
        space. Stars-and-bars argument applies here.

        :param n: Number of particles.
        :param m: Number of modes.
        :return: Dimension of n-particle m-mode bosonic space.
    """
    return round(binom(n + m - 1, n))


def calculate_number_of_possible_lossy_n_particle_m_mode_output_states(
    n: int, m: int
) -> int:
    """
        Calculates the number of possible output states with N <= n particles placed
        around m modes.

        :param n: Number of particles.
        :param m: Number of modes.
        :return: Dimension of n-particle m-mode bosonic space.
    """
    states_number = 0
    for N in range(n + 1):
        states_number += round(binom(N + m - 1, N))
    return states_number


def get_modes_transmissivity_values_from_matrix(
    lossy_interferometer_matrix: ndarray,
) -> List[float]:
    v_matrix, singular_values, u_matrix = svd(lossy_interferometer_matrix)
    return square(flip(singular_values))


def _calculate_singular_values_matrix_expansion(
    singular_values_vector: ndarray,
) -> ndarray:
    vector_of_squared_expansions = 1.0 - power(singular_values_vector, 2)
    for i in range(len(vector_of_squared_expansions)):
        if vector_of_squared_expansions[i] < 0:
            vector_of_squared_expansions[i] = 0

    expansion_values = sqrt(vector_of_squared_expansions)

    return diag(expansion_values)


def prepare_interferometer_matrix_in_expanded_space(
    interferometer_matrix: Sequence[Sequence[complex128]],
) -> ndarray:
    v_matrix, singular_values, u_matrix = svd(interferometer_matrix)

    expansions_zeros = zeros_like(v_matrix)
    expansions_ones = eye(len(v_matrix))
    expanded_v = block(
        [[v_matrix, expansions_zeros], [expansions_zeros, expansions_ones]]
    )
    expanded_u = block(
        [[u_matrix, expansions_zeros], [expansions_zeros, expansions_ones]]
    )
    singular_values_matrix_expansion = _calculate_singular_values_matrix_expansion(
        singular_values
    )
    singular_values_expanded_matrix = block(
        [
            [diag(singular_values), singular_values_matrix_expansion],
            [singular_values_matrix_expansion, diag(singular_values)],
        ]
    )
    return expanded_v @ singular_values_expanded_matrix @ expanded_u


def compute_state_types(
    modes_number: int, particles_number: int, losses: bool = False
) -> List[List[int]]:
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

    for i in range(len(state_types)):
        while len(state_types[i]) < modes_number:
            state_types[i].append(0)

    return state_types


def compute_number_of_state_types(
    modes_number: int, particles_number: int, losses=False
) -> int:
    state_types_number = 0

    for k in range(1, modes_number + 1):
        state_types_number += compute_number_of_k_element_integer_n_partitions(
            k, particles_number
        )

    if not losses:
        return state_types_number

    for particles_num in range(particles_number):
        for k in range(1, modes_number + 1):
            state_types_number += compute_number_of_k_element_integer_n_partitions(
                k, particles_num
            )

    return state_types_number


def compute_number_of_k_element_integer_n_partitions(k: int, n: int) -> int:
    if k == 1:
        return 1

    if k > n or n == 0 or k < 1:
        return 0

    integer_partitions_number = compute_number_of_k_element_integer_n_partitions(
        k, n - k
    )
    integer_partitions_number += compute_number_of_k_element_integer_n_partitions(
        k - 1, n - 1
    )

    return integer_partitions_number


def compute_maximally_unbalanced_types(
    modes_number: int, particles_number: int
) -> List[List[int]]:
    maximally_unbalanced_types = []
    all_types = compute_state_types(
        particles_number=particles_number, modes_number=modes_number
    )

    for state_type in all_types:
        if state_type.count(1) == len(state_type) - 1 or state_type.count(1) == len(
            state_type
        ):
            maximally_unbalanced_types.append(state_type)

    return maximally_unbalanced_types


def compute_number_of_states_of_given_type(state_type: Sequence[int]) -> int:
    modes_number = len(state_type)

    counts = []
    vals = set(state_type)

    for val in vals:
        counts.append(state_type.count(val))

    type_count = factorial(modes_number)

    for count in counts:
        type_count /= factorial(count)

    number_of_states_of_given_type = factorial(modes_number)

    for count in counts:
        number_of_states_of_given_type //= factorial(count)

    return number_of_states_of_given_type


def compute_state_of_given_type(state_type: Sequence[int]) -> List[Tuple[int]]:

    if len(state_type) == 0:
        return [tuple()]

    states_of_type: Set[Tuple[int, ...]] = set()

    working_state: List[int] = list(state_type)
    modes_number: int = len(state_type)

    for _ in range(modes_number):
        val: int = working_state.pop(0)
        substates: List[Tuple[int]] = compute_state_of_given_type(working_state)

        for substate in substates:
            states_of_type.add((val,) + substate)

        working_state.append(val)

    return list(states_of_type)


def generate_qft_matrix_for_first_m_modes(m: int, all_modes_number: int) -> ndarray:
    small_qft_matrix = compute_qft_matrix(m)
    qft_matrix = eye(all_modes_number, dtype=complex128)
    qft_matrix[0:m, 0:m] = small_qft_matrix
    return qft_matrix


def generate_random_phases_matrix_for_first_m_modes(
    m: int, all_modes_number: int
) -> ndarray:
    random_phases = ones(all_modes_number, dtype=complex128)  # [1, 1, 1, 1, 1, 1]
    random_phases[0:m] = exp(1j * 2 * pi * rand(m))
    return diag(random_phases)


class EffectiveScatteringMatrixCalculator:
    """
        In many methods of Boson Sampling simulations an effective scattering matrix has
        to be calculated. Therefore, I decided to implement a calculator that'd be used
        in every single one of these methods.
    """

    def __init__(
        self,
        matrix: Sequence[Sequence[complex128]],
        input_state: Optional[Sequence[int]] = None,
        output_state: Optional[Sequence[int]] = None,
    ) -> None:
        if output_state is None:
            output_state = list()
        if input_state is None:
            input_state = list()
        self._matrix: Sequence[Sequence[complex128]] = matrix
        self._input_state: Sequence[int] = input_state
        self._output_state: Sequence[int] = output_state

    @property
    def matrix(self) -> Sequence[Sequence[complex128]]:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: Sequence[Sequence[complex128]]) -> None:
        self._matrix = matrix

    @property
    def input_state(self) -> Sequence[int]:
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: Sequence[int]) -> None:
        self._input_state = input_state

    @property
    def output_state(self) -> Sequence[int]:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: Sequence[int]) -> None:
        self._output_state = output_state

    def calculate(self) -> Sequence[Sequence[complex128]]:

        if sum(self.input_state) == 0 or sum(self.output_state) == 0:
            return []

        transposed_input_matrix = transpose(self._matrix)
        helper_mtx = []

        for index_of_column_to_insert in range(len(self._input_state)):
            helper_mtx += [transposed_input_matrix[index_of_column_to_insert]] * int(
                self._input_state[index_of_column_to_insert]
            )

        helper_mtx = transpose(array(helper_mtx, dtype=complex128))

        effective_scattering_matrix = []

        for index_of_row_to_insert in range(len(self._output_state)):
            effective_scattering_matrix += [helper_mtx[index_of_row_to_insert]] * int(
                self._output_state[index_of_row_to_insert]
            )

        return effective_scattering_matrix
