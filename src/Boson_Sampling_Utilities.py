__author__ = 'Tomasz Rybotycki'

# TR TODO: Consider making this file a package along with exact distribution calculator.

import itertools
from typing import List, Optional

from numpy import array, block, complex128, diag, ndarray, ones_like, power, sqrt, zeros, zeros_like
from numpy.linalg import svd
from scipy.special import binom


def calculate_permanent(matrix: ndarray) -> complex128:
    """
    Returns the permanent of the matrix.
    """
    return permanent_recursive_part(matrix, column=0, selected=array([]), prod=complex128(1))


def permanent_recursive_part(mtx: ndarray, column: int, selected: ndarray, prod: complex128) -> complex128:
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column,
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        return prod

    result = complex128(0 + 0j)
    for row in range(mtx.shape[0]):
        if row not in selected:
            result += permanent_recursive_part(mtx, column + 1, selected + [row], prod * mtx[row, column])
    return result


def particle_state_to_modes_state(particle_state: ndarray, observed_modes_number: int) -> ndarray:
    modes_state = zeros(observed_modes_number)

    # Adding the particle to it's mode.
    for particle in particle_state:
        # I'd expect particle_state would ideally be an array of ints, but after some transformations it may be
        # parsed to floats, thus casting.
        modes_state[int(particle)] += 1

    return modes_state


def modes_state_to_particle_state(mode_state: ndarray, particles_number: int) -> ndarray:
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


def generate_possible_outputs(number_of_particles: int, number_of_modes: int) -> List[ndarray]:
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

    return outputs


def generate_lossy_inputs(initial_state: ndarray, number_of_particles_left: int) -> List[ndarray]:
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

    # Symmetrization.
    for combination in itertools.combinations(list(range(initial_number_of_particles)), number_of_particles_left):
        lossy_input_in_particle_basis = array([x0[el] for el in combination])

        lossy_input = particle_state_to_modes_state(lossy_input_in_particle_basis, number_of_modes)

        # Check if calculated lossy input is already in the list. If not, add it.
        lossy_inputs_ids = map(id, lossy_inputs_list)

        if id(lossy_input) not in lossy_inputs_ids:
            lossy_inputs_list.append(lossy_input)

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
    return binom(n + m - 1, n)


def prepare_interferometer_matrix_in_expanded_space(interferometer_matrix: ndarray) -> ndarray:
    v_matrix, singular_values, u_matrix = svd(interferometer_matrix)
    expansions_zeros = zeros_like(v_matrix)
    expansions_ones = ones_like(v_matrix)
    expanded_v = block([[v_matrix, expansions_zeros], [expansions_zeros, expansions_ones]])
    expanded_u = block([[u_matrix, expansions_zeros], [expansions_zeros, expansions_ones]])
    singular_values_matrix_expansion = _calculate_singular_values_matrix_expansion(singular_values)
    singular_values_expanded_matrix = block([[diag(singular_values), singular_values_matrix_expansion],
                                             [singular_values_matrix_expansion, diag(singular_values)]])
    return expanded_v @ singular_values_expanded_matrix @ expanded_u


def _calculate_singular_values_matrix_expansion(singular_values_vector: ndarray) -> ndarray:
    expansion_values = sqrt(1.0 - power(singular_values_vector, 2))
    return diag(expansion_values)


class EffectiveScatteringMatrixCalculator:
    """
        In many methods of Boson Sampling simulations an effective scattering matrix has to be calculated. Therefore
        I decided to implement an calculator that'd be used if every single one of these methods.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        if output_state is None:
            output_state = array([])
        if input_state is None:
            input_state = array([])
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
        self.__input_state = input_state

    @property
    def output_state(self) -> ndarray:
        return self.__output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self.__output_state = output_state

    def calculate(self) -> ndarray:
        number_of_columns = sum(self.input_state)
        effective_scattering_matrix = zeros(shape=(number_of_columns, number_of_columns), dtype=complex128)
        helper_matrix = zeros(shape=(len(self.__matrix), number_of_columns), dtype=complex128)
        next_column_index = 0

        for j in range(len(self.input_state)):
            for i in range(self.input_state[j]):
                helper_matrix[:, [next_column_index]] = self.__matrix[:, [j]]
                next_column_index += 1
        next_row_index = 0

        for j in range(len(self.output_state)):
            for i in range(int(self.output_state[j])):
                effective_scattering_matrix[[next_row_index], :] = helper_matrix[[j], :]

                next_row_index += 1

        return effective_scattering_matrix


class EffectiveScatteringMatrixPermanentCalculator:
    """
        This class is used to calculate permanent of effective scattering matrix. It first
        generates the matrix, and then calculates the permanent via standard means.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        if output_state is None:
            output_state = array([])
        if input_state is None:
            input_state = array([])
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
        self.__input_state = input_state

    @property
    def output_state(self) -> ndarray:
        return self.__output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self.__output_state = output_state

    def calculate(self) -> complex128:
        scattering_matrix_calculator = \
            EffectiveScatteringMatrixCalculator(self.__matrix, self.__input_state, self.__output_state)
        scattering_matrix = scattering_matrix_calculator.calculate()
        return calculate_permanent(scattering_matrix)


class ChinHuhPermanentCalculator:
    """
        This class is designed to calculate permanent of effective scattering matrix of a boson sampling instance.
        Note, that it can be used to calculate permanent of given matrix. All that is required that input and output
        states are set to [1, 1, ..., 1] with proper dimensions.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        if output_state is None:
            output_state = []
        if input_state is None:
            input_state = []
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
        self.__input_state = input_state

    @property
    def output_state(self) -> ndarray:
        return self.__output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self.__output_state = output_state

    def calculate(self) -> complex128:
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """
        if not self.__can_calculation_be_performed():
            raise AttributeError

        v_vectors = self.__calculate_v_vectors()

        permanent = complex128(0)
        for v_vector in v_vectors:
            v_sum = sum(v_vector)
            addend = pow(-1, v_sum)
            # Binomials calculation
            for i in range(len(v_vector)):
                addend *= binom(self.__input_state[i], v_vector[i])
            # Product calculation
            product = 1
            for j in range(len(self.__input_state)):
                if self.__output_state[j] == 0:  # There's no reason to calculate the sum if t_j = 0
                    continue
                # Otherwise we calculate the sum
                product_part = 0
                for i in range(len(self.__input_state)):
                    product_part += (self.__input_state[i] - 2 * v_vector[i]) * self.__matrix[j][i]
                product_part = pow(product_part, self.__output_state[j])
                product *= product_part
            addend *= product
            permanent += addend
        permanent /= pow(2, sum(self.__input_state))
        return permanent

    def __can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self.__matrix.shape[0] == self.__matrix.shape[1] \
            and len(self.__output_state) == len(self.__input_state) \
            and len(self.__output_state) == self.__matrix.shape[0]

    def __calculate_v_vectors(self, input_vector: Optional[ndarray] = None) -> List[ndarray]:
        if input_vector is None:
            input_vector = []
        v_vectors = []
        for i in range(self.__input_state[len(input_vector)] + 1):
            input_state = input_vector.copy()
            input_state.append(i)

            if len(input_state) == len(self.__input_state):
                v_vectors.append(input_state)
            else:
                v_vectors.extend(self.__calculate_v_vectors(input_state))

        return v_vectors
