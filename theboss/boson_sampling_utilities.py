__author__ = "Tomasz Rybotycki"

"""
    This script contains various auxiliary methods useful for boson sampling
    experiments.
"""

import itertools
from typing import List, Optional, Sequence, Tuple

from numpy import (
    ndarray,
    array,
    block,
    complex128,
    diag,
    eye,
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
from numpy.random import rand, random

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
        observed_modes_number = max(modes_assignment) + 1

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
            mode_assignment += (i,)

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
        return list()
    if particles_number == 0:
        return [tuple([0 for _ in range(modes_number)])]

    states = []
    starting_number_of_particles = particles_number

    if losses:
        starting_number_of_particles = 0

    for particles_number in range(starting_number_of_particles, particles_number + 1):
        states.extend(
            _generate_possible_n_particle_states(particles_number, modes_number)
        )

    return states


def _generate_possible_n_particle_states(n: int, modes_number: int) -> List[Tuple[int]]:
    """
    Generates all possible :math:`n` particle states.

    :param n:
        The number of particles in resultant states.
    :param modes_number:
        The number :math:`m` of modes in resultant states.

    :return:
        A list of possible :math:`m`-mode `n`-particle states in 2nd quantization
        representation (as tuples of ints).
    """
    states = []

    state: List[int] = [0 for _ in range(modes_number)]

    state[0] = n
    states.append(state)

    while states[-1][modes_number - 1] < n:

        k = modes_number - 1
        while states[-1][k - 1] == 0:
            k -= 1

        state = states[-1].copy()
        state[k - 1] -= 1

        for i in range(k, len(state)):
            state[i] = 0

        state[k] = n - sum(state)

        states.append(state)

    sorted_states = sorted([tuple(output) for output in states], reverse=True)

    return sorted_states


def generate_lossy_n_particle_input_states(
    initial_state: Sequence[int], number_of_particles_left: int
) -> List[Tuple[int]]:
    """
    From initial state generate all possible input states, with required number of
    particles after losses application.

    Notice that it can also be done using the Guan Codes!

    :param initial_state:
        The state we start with.
    :param number_of_particles_left:
        Number of particles after losses application.

    :return:
        A list of tuples of ints representing initial states after losses.
    """
    x0 = []
    number_of_modes = len(initial_state)
    initial_number_of_particles = sum(initial_state)
    for i in range(number_of_modes):
        x0.extend([i] * int(initial_state[i]))

    lossy_inputs_list = []
    lossy_inputs_hashes = []

    if sum(initial_state) == 0:
        return [tuple(initial_state)]

    # Symmetrization.
    for combination in itertools.combinations(
        list(range(initial_number_of_particles)), number_of_particles_left
    ):
        lossy_input_in_particle_basis = array([x0[el] for el in combination], dtype=int)

        lossy_input = mode_assignment_to_mode_occupation(
            lossy_input_in_particle_basis, number_of_modes
        )

        # Check if calculated lossy input is already in the list. If not, add it.
        lossy_input_hash = hash(tuple(lossy_input))
        if lossy_input_hash not in lossy_inputs_hashes:
            lossy_inputs_list.append(lossy_input)
            lossy_inputs_hashes.append(lossy_input_hash)

    return lossy_inputs_list


def bosonic_space_dimension(
    particles_number: int, modes_number: int, losses: bool = False
) -> int:
    """
    Calculates the number of possible states with specified number of modes and
    maximal number of particles.

    This is basically the same answer as to in how many possible combinations can we
    put :math:`n` objects in :math:`m` bins. It's also a dimension of :math:`m`-mode
    bosonic space with at most :math:`n` particles, or exactly :math:`n` if we don't
    consider losses. Stars-and-bars argument applies here.

    :param particles_number:
        Number :math:`n` of particles. If lossy states are considered, this is the
        maximal number of particles.
    :param modes_number:
        Number :math:`m` of considered modes.

    :return:
        Dimension of (possibly lossy) :math:`m`-mode bosonic space with at most
        :math:`n` particles.
    """

    dimension: int = round(binom(particles_number + modes_number - 1, particles_number))

    if not losses:
        return dimension

    for n in range(particles_number):
        dimension += round(binom(n + modes_number - 1, n))

    return dimension


def get_modes_transmission_probabilities_from_matrix(
    lossy_interferometer_matrix: Sequence[Sequence[complex128]],
) -> List[float]:
    """
    Given a lossy interferometer matrix this method extracts from it the information
    about the transmittance of the modes. Given that SVD decomposition is not unique
    the returned order of the transmittance values doesn't correspond to the order on
    modes in general.

    It so happens that the transmission probabilities in the lossy interferometer matrix
    are described by the roots of the singular values (in our case the eigenvalues of
    the matrix).

    :param lossy_interferometer_matrix:
        A lossy interferometer matrix from which the information about the
        transmission probabilities will be extracted.

    :return:
        Unordered list of modes transmission probabilities extracted from given matrix.
    """
    v_matrix, singular_values, u_matrix = svd(lossy_interferometer_matrix)
    return square(flip(singular_values))


def _compute_loss_transfer_matrix_expansion(transmittances: ndarray,) -> ndarray:
    """
    Returns extension part of the singular values' matrix resulting from the SVD
    decomposition of the (presumably lossy) interferometer.

    :param transmittances:
        The values of transmittances obtained from the squares of the singular values
        of the (presumably lossy) interferometer matrix' SVD.

    :return:
        One of the block matrices of singular values' matrix of the SVD of the given
        (presumably lossy) interferometer in expanded space.
    """
    losses_vector = 1.0 - transmittances
    for i in range(len(losses_vector)):
        if losses_vector[i] < 0:
            losses_vector[i] = 0

    expansion_values = sqrt(losses_vector)

    return diag(expansion_values)


def prepare_interferometer_matrix_in_expanded_space(
    interferometer_matrix: Sequence[Sequence[complex]],
) -> ndarray:
    """
    This operation is required for the simulation of BS experiment with mode dependent
    (non-uniform) losses.

    One way to perform such simulation is to expand the experiment from the
    :math:`m \\times m` to :math:`2m \\times 2m` one and treat the additional
    modes as the space for the lost particles. Then the loss of a particle can be
    implemented as transferring it to one of the additional modes. By the end of the
    simulation the additional modes have to be trimmed.

    Notice that this is not necessary in the case of uniform losses, but can also be
    used for it.

    Although it's not necessary, the method returns a unitary matrix.

    :param interferometer_matrix:
        An (possibly lossy) interferometer matrix to be expanded.

    :return:
        Given interferometer in the expanded sampling space.
    """
    v_matrix, transmittances, u_matrix = svd(interferometer_matrix)

    extension_zeros_matrix = zeros_like(v_matrix)
    extension_identity_matrix = eye(len(v_matrix))

    expanded_v = block(
        [
            [v_matrix, extension_zeros_matrix],
            [extension_zeros_matrix, extension_identity_matrix],
        ]
    )

    expanded_u = block(
        [
            [u_matrix, extension_zeros_matrix],
            [extension_zeros_matrix, extension_identity_matrix],
        ]
    )

    transmission_probabilities = array([s ** 2 for s in transmittances])
    loss_transfer_extension_matrix = _compute_loss_transfer_matrix_expansion(
        transmission_probabilities
    )

    # This is the most specific thing here.
    expanded_singular_values_matrix = block(
        [
            [diag(transmittances), loss_transfer_extension_matrix],
            [loss_transfer_extension_matrix, diag(transmittances)],
        ]
    )
    return expanded_v @ expanded_singular_values_matrix @ expanded_u


def generate_state_types(
    modes_number: int, particles_number: int, losses: bool = False
) -> List[Tuple[int, ...]]:
    """
    Returns a list of (possibly lossy) state types understood in the same sense as
    in [1]. We also assume that the modes occupations of the state types are ordered
    in the non-increasing order, as in [1].

    :param modes_number:
        The number :math:`m` of considered modes.
    :param particles_number:
        The maximal number :math:`n` of considered
    :param losses:
        A flag informing if losses should be considered.
    :return:
        A list of (possibly lossy) state types given by the lists of ints.
    """

    def _partitions(n, I=1):
        """
        A method for generating integer partitions.
        Credits to
        https://stackoverflow.com/questions/10035752/elegant-python-code-for-integer-partitioning/10036764

        :param n:
            The number for which integer partitions will be returned.
        :param I:
            A control parameter.
        :return:
            The integer partitions of :math:`n`.
        """
        yield (n,)
        for i in range(I, n // 2 + 1):
            for p in _partitions(n - i, i):
                yield (i,) + p

    all_partitions = list(_partitions(particles_number))

    if losses:
        for i in range(particles_number):
            all_partitions += list(_partitions(i))

    state_types = []

    for partition in all_partitions:
        if len(partition) > modes_number:
            continue
        # We describe state type by a vector in descending order [1].
        state_type = sorted(partition, reverse=True)
        state_types.append(state_type)

    for i in range(len(state_types)):
        while len(state_types[i]) < modes_number:
            state_types[i].append(0)

    return [tuple(state_type) for state_type in state_types]


def compute_number_of_state_types(
    modes_number: int, particles_number: int, losses=False
) -> int:
    """
    Computes the number of state types (as defined in [1]) with given number of modes
    and particles. It also allows the case when the losses are considered.

    The number of state types, given modes number :math:`m` and particles number
    :math:`n` is equal to the number of integer partitions of :math:`n` of length
    at most :math:`m`. In case if losses are allowed one has to sum up the number of
    partitions for numbers :math:`p` of particles such that :math:`0 \\leq p \\leq n`.

    :param modes_number:
        The number :math:`m` of considered modes.
    :param particles_number:
        The maximal number :math:`n` of allowed particles.
    :param losses:
        A flag indicating whether the lossy states should also be considered.
    :return:
        The number of (possibly lossy) state types for given number of modes and
        particles.
    """
    state_types_number = 0

    for k in range(1, modes_number + 1):
        state_types_number += compute_number_of_k_element_integer_partitions_of_n(
            k, particles_number
        )

    if not losses:
        return state_types_number

    for particles_num in range(particles_number):
        for k in range(1, modes_number + 1):
            state_types_number += compute_number_of_k_element_integer_partitions_of_n(
                k, particles_num
            )

    return state_types_number


def compute_number_of_states_of_given_type(state_type: Sequence[int]) -> int:
    """
    Returns the number of possible states of given type. The two states are of the same
    type if they can be mapped into each other by a mode-permuting matrix. The number
    of states of possible type is therefore the number of distinct permutations of given
    state type.

    :param state_type:
        A state type as defined in [1]. Multiple states can be of the same type so
        this can also be understood as a representative of desired state type.
    :return:
        The number of states of given type.
    """
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


def compute_number_of_k_element_integer_partitions_of_n(k: int, n: int) -> int:
    """
    Return the number of :math:`k`-element partitions of integer :math:`n`.

    :param k:
        The size of the partitions.
    :param n:
        The number for which the number of :math:`k`-element partitions is computed.
    :return:
        The number of :math:`k`-element partitions of :math:`n`.
    """
    if k == 1:
        return 1

    if k > n or n == 0 or k < 1:
        return 0

    integer_partitions_number = compute_number_of_k_element_integer_partitions_of_n(
        k, n - k
    )
    integer_partitions_number += compute_number_of_k_element_integer_partitions_of_n(
        k - 1, n - 1
    )

    return integer_partitions_number


def generate_qft_matrix_for_first_m_modes(m: int, all_modes_number: int) -> ndarray:
    """
    Prepares a matrix which describes a gate that applies QFT on the first :math:`m`
    modes and identity on the rest.

    :param m:
        The number of initial modes on which the QFT will be applied.
    :param all_modes_number:
        The number of all the modes and, consequently, the size of the resultant matrix.
    :return:
        A square matrix of a size given by the number of all modes which applies QFT
        on the first :math:`m` modes.
    """
    small_qft_matrix = compute_qft_matrix(m)
    qft_matrix = eye(all_modes_number, dtype=complex128)
    qft_matrix[0:m, 0:m] = small_qft_matrix
    return qft_matrix


def generate_random_phases_matrix_for_first_m_modes(
    m: int, all_modes_number: int
) -> ndarray:
    """
    Returns a matrix that applies random phases on the first :math:`m` modes and the
    identity on all the others.

    :param m:
        The number of initial modes on which the random phases should be applied.
    :param all_modes_number:
        The total number of considered modes and, consequently, the size of the
        resultant matrix.
    :return:
        A matrix describing an operation of applying random phases on the first
        :math:`m` modes and the identity on all the others.
    """
    random_phases = ones(all_modes_number, dtype=complex128)  # [1, 1, 1, 1, 1, 1]
    random_phases[0:m] = exp(1j * 2 * pi * rand(m))
    return diag(random_phases)


# TODO TR:  Ensure that
def apply_uniform_losses_to_the_state(
    state: Sequence[int], transmission_probability: float
) -> Tuple[int, ...]:
    """
    Applies uniform losses to the given state.

    :param state:
        State to which uniform losses will be applied.
    :param transmission_probability:
        Uniform probability of boson transmission.

    :return:
        Lossy input state.
    """
    lossy_input = [0 for _ in state]

    for mode in range(len(state)):
        for particle in range(state[mode]):
            if random() <= transmission_probability:
                lossy_input[mode] += 1

    return tuple(lossy_input)


# TODO TR:  This is possibly use in many places. Find these places and use this method
#           instead.
def compute_binomial_weights(
    total_particles_number: int, transmission_probability: float
) -> List[float]:
    """
    Computes the binomial weights for a given, uniformly lossy, BS experiment instance.

    :param total_particles_number:
        The initial number of particles in the input state.
    :param transmission_probability:
        The uniform probability of particle transmission through the network.

    :return:
        The binomial weights describing the probabilities of loosing a number of
        particles specified by the index.
    """
    weights: List[float] = []

    def binomial_weight(n: int, l: int, eta: float) -> float:
        return pow(eta, l) * pow(1 - eta, n - l) * binom(n, l)

    for particles_left in range(total_particles_number + 1):
        weights.append(
            binomial_weight(
                total_particles_number, particles_left, transmission_probability
            )
        )

    return weights


def generate_standard_state(
    modes_number: int, particles_number: int
) -> Tuple[int, ...]:
    """
    Creates a :math:`m`-mode :math:`n`-particle standard Fock input state, which is
    in form [1, 1, ..., 1, 0, 0, ... 0].

    :param modes_number:
        The number of modes :math:`m` of the resultant state.
    :param particles_number:
        The number of particles :math:`n`

    :return:
        An :math:`m`-mode :math:`n`-particle standard Fock input state.
    """
    standard_state: List[int] = [0 for _ in range(modes_number)]
    standard_state[0:particles_number] = [1 for _ in range(particles_number)]
    return tuple(standard_state)


class EffectiveScatteringMatrixCalculator:
    """
    In many methods of Boson Sampling simulations an effective scattering matrix has
    to be calculated. Therefore, I decided to implement a calculator that'd be used
    in every single one of these methods.

    For the method to work properly the input and the output states should both
    be provided in the 2nd quantization representation (mode occupation).
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
        # Note that we expect 2nd quantization description here.
        self._input_state = input_state

    @property
    def output_state(self) -> Sequence[int]:
        # Note that we expect 2nd quantization description here.
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: Sequence[int]) -> None:
        self._output_state = output_state

    def calculate(self) -> List[List[complex128]]:
        """
        Calculates and returns the effective scattering matrix in the BS instance
        for previously given input state, output state and the interferometer matrix.

        Note that for the proper results we expect input state and the output state
        to be in the 2nd quantization representation.

        :return:
            The effective scattering matrix in the specified BS instance.
        """
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
