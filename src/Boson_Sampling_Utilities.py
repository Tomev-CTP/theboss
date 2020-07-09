__author__ = 'Tomasz Rybotycki'

# TR TODO: Consider making this file a package along with exact distribution calculator.

import itertools
from typing import List
from numpy import ndarray, zeros


def calculate_permanent(matrix: ndarray) -> float:
    """
    Returns the permanent of the matrix mat.
    """
    return permanent_recursive_part(matrix, 0, [], 1)


def permanent_recursive_part(mtx: ndarray, column: int, selected: List[int], prod: int) -> float:
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column,
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        return prod

    result = 0
    for row in range(mtx.shape[0]):
        if row not in selected:
            result = result \
                     + permanent_recursive_part(mtx, column + 1, selected + [row], prod * mtx[row, column])
    return result


def particle_state_to_modes_state(particle_state: ndarray, observed_modes_number: int) -> ndarray:
    modes_state = zeros(observed_modes_number)

    # Adding the particle to it's mode.
    for particle in particle_state:
        modes_state[int(particle)] += 1

    # numbers of particles in each mode
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
            particles_state[k] = int(i)
            k += 1
        else:
            i += 1

    return particles_state


def generate_possible_outputs(number_of_particles: int, number_of_modes: int) -> List[ndarray]:
    outputs = []

    output = zeros(number_of_modes)
    output[0] = number_of_particles
    outputs.append(output)

    while outputs[-1][number_of_modes - 1] != number_of_particles:

        k = number_of_modes - 1
        while outputs[-1][k - 1] == 0:
            k -= 1

        output = outputs[-1].copy()
        output[k - 1] -= 1
        output[k:] = 0
        output[k] = number_of_particles - sum(output)

        outputs.append(output)

    return outputs


def generate_lossy_inputs(initial_state: ndarray, number_of_particles_left: int) -> List[List[int]]:
    """
    From initial state generate all possible input states after losses application.
    :param initial_state: The state we start with.
    :param number_of_particles_left: Number of particles after losses application.
    :return: A list of lists representing initial states after losses.
    """
    x0 = []
    number_of_modes = len(initial_state)
    initial_number_of_particles = len(initial_state) - initial_state.count(0)
    for i in range(number_of_modes):
        x0.extend([i] * int(initial_state[i]))

    lossy_inputs_list = []

    # Symmetrization
    for combination in itertools.combinations(list(range(initial_number_of_particles)), number_of_particles_left):
        lossy_input_in_particle_basis = []
        for el in combination:
            lossy_input_in_particle_basis.append(x0[el])

        lossy_input = particle_state_to_modes_state(lossy_input_in_particle_basis, number_of_modes)

        # Check if calculated lossy input is already in the list. If not, add it.
        if all(list(lossy_input_in_list) != list(lossy_input) for lossy_input_in_list in lossy_inputs_list):
            lossy_inputs_list.append(lossy_input)

    return lossy_inputs_list
