__author__ = 'Tomasz Rybotycki'

# TR TODO: Consider making this file a package along with exact distribution calculator.

import itertools
import multiprocessing
from typing import List, Optional

from joblib import delayed, Parallel
from numpy import array, asarray, block, complex128, diag, eye, int64, ndarray, power, sqrt, transpose, zeros, \
    zeros_like
from numpy.linalg import svd
from scipy.special import binom


def calculate_permanent(matrix: ndarray) -> complex128:
    """
    Returns the permanent of the matrix.
    """
    return permanent_recursive_part(matrix, column=0, selected=[], prod=complex128(1))


def permanent_recursive_part(mtx: ndarray, column: int, selected: List[int], prod: complex128) -> complex128:
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
    for particles_mode in asarray(particle_state, dtype=int64):
        modes_state[particles_mode] += 1

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
    lossy_inputs_hashes = []

    if sum(initial_state) == 0:
        return [initial_state]

    # Symmetrization.
    for combination in itertools.combinations(list(range(initial_number_of_particles)), number_of_particles_left):
        lossy_input_in_particle_basis = array([x0[el] for el in combination], dtype=int64)

        lossy_input = particle_state_to_modes_state(lossy_input_in_particle_basis, number_of_modes)

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
    return binom(n + m - 1, n)


def prepare_interferometer_matrix_in_expanded_space(interferometer_matrix: ndarray) -> ndarray:
    v_matrix, singular_values, u_matrix = svd(interferometer_matrix)
    expansions_zeros = zeros_like(v_matrix)
    expansions_ones = eye(len(v_matrix))
    expanded_v = block([[v_matrix, expansions_zeros], [expansions_zeros, expansions_ones]])
    expanded_u = block([[u_matrix, expansions_zeros], [expansions_zeros, expansions_ones]])
    singular_values_matrix_expansion = _calculate_singular_values_matrix_expansion(singular_values)
    singular_values_expanded_matrix = block([[diag(singular_values), singular_values_matrix_expansion],
                                             [singular_values_matrix_expansion, diag(singular_values)]])
    return expanded_v @ singular_values_expanded_matrix @ expanded_u


def _calculate_singular_values_matrix_expansion(singular_values_vector: ndarray) -> ndarray:
    # return diag(1.0 - singular_values_vector)

    vector_of_squared_expansions = 1.0 - power(singular_values_vector, 2)
    for i in range(len(vector_of_squared_expansions)):
        if vector_of_squared_expansions[i] < 0:
            vector_of_squared_expansions[i] = 0

    expansion_values = sqrt(vector_of_squared_expansions)

    return diag(expansion_values)


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
                int(self.__output_state[index_of_row_to_insert])

        return array(effective_scattering_matrix, dtype=complex128)


class EffectiveScatteringMatrixPermanentCalculator:
    """
        This class is used to calculate permanent of effective scattering matrix. It first
        generates the matrix, and then calculates the permanent via standard means.
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
            output_state = array([], dtype=int64)
        if input_state is None:
            input_state = array([], dtype=int64)
        self._matrix = matrix
        self._input_state = input_state
        self._output_state = output_state

    @property
    def matrix(self) -> ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix: ndarray) -> None:
        self._matrix = matrix

    @property
    def input_state(self) -> ndarray:
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: ndarray) -> None:
        self._input_state = asarray(input_state, dtype=int64)

    @property
    def output_state(self) -> ndarray:
        return self._output_state

    @output_state.setter
    def output_state(self, output_state: ndarray) -> None:
        self._output_state = asarray(output_state, dtype=int64)

    def calculate(self) -> complex128:
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """
        if not self._can_calculation_be_performed():
            raise AttributeError

        v_vectors = self._calculate_v_vectors()

        permanent = complex128(0)
        for v_vector in v_vectors:
            permanent += self.compute_permanent_addend(v_vector)
        permanent /= pow(2, sum(self._input_state))
        return permanent

    def _can_calculation_be_performed(self) -> bool:
        """
            Checks if calculation can be performed. For this to happen sizes of given matrix and states have
            to match.
            :return: Information if the calculation can be performed.
        """
        return self._matrix.shape[0] == self._matrix.shape[1] \
               and len(self._output_state) == len(self._input_state) \
               and len(self._output_state) == self._matrix.shape[0]

    def _calculate_v_vectors(self, input_vector: Optional[ndarray] = None) -> List[ndarray]:
        if input_vector is None:
            input_vector = []
        v_vectors = []
        for i in range(int(self._input_state[len(input_vector)]) + 1):
            input_state = input_vector.copy()
            input_state.append(i)

            if len(input_state) == len(self._input_state):
                v_vectors.append(input_state)
            else:
                v_vectors.extend(self._calculate_v_vectors(input_state))

        return v_vectors

    def compute_permanent_addend(self, v_vector: ndarray) -> complex128:
        v_sum = sum(v_vector)
        addend = pow(-1, v_sum)
        # Binomials calculation
        for i in range(len(v_vector)):
            addend *= binom(self._input_state[i], v_vector[i])
        # Product calculation
        product = 1
        for j in range(len(self._input_state)):
            if self._output_state[j] == 0:  # There's no reason to calculate the sum if t_j = 0
                continue
            # Otherwise we calculate the sum
            product_part = 0
            for i in range(len(self._input_state)):
                product_part += (self._input_state[i] - 2 * v_vector[i]) * self._matrix[j][i]
            product_part = pow(product_part, self._output_state[j])
            product *= product_part
        addend *= product
        return addend


class ParallelChinHuhPermanentCalculator(ChinHuhPermanentCalculator):
    """
        This class is meant to parallelize the CH Permanent Calculator.
    """

    def __init__(self, matrix: ndarray, input_state: Optional[ndarray] = None,
                 output_state: Optional[ndarray] = None) -> None:
        super().__init__(matrix, input_state, output_state)

    def calculate(self) -> complex128:
        """
            This is the main method of the calculator. Assuming that input state, output state and the matrix are
            defined correctly (that is we've got m x m matrix, and vectors of with length m) this calculates the
            permanent of an effective scattering matrix related to probability of obtaining output state from given
            input state.
            :return: Permanent of effective scattering matrix.
        """
        if not self._can_calculation_be_performed():
            raise AttributeError

        v_vectors = self._calculate_v_vectors()

        permanent = complex128(0)

        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self.compute_permanent_addend)(v_vector) for v_vector in v_vectors)
        permanent += sum(results)
        permanent /= pow(2, sum(self._input_state))

        return permanent
