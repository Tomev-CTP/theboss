__author__ = 'Tomasz Rybotycki'

# TR TODO: Add proper type hints.
import numpy as np


def permanent_recursive_part(mtx, column, selected, prod):
    """
    Row expansion for the permanent of matrix mtx.
    The counter column is the current column,
    selected is a list of indices of selected rows,
    and prod accumulates the current product.
    """
    if column == mtx.shape[1]:
        return prod
    else:
        result = 0
        for row in range(mtx.shape[0]):
            if row not in selected:
                result = result \
                         + permanent_recursive_part(mtx, column + 1, selected + [row], prod * mtx[row, column])
        return result


def calculate_permanent(mat):
    """
    Returns the permanent of the matrix mat.
    """
    return permanent_recursive_part(mat, 0, [], 1)


def particle_state_to_modes_state(particle_state: np.ndarray, observed_modes_number: int) -> np.ndarray:
    modes_state = np.zeros(observed_modes_number)

    # Adding the particle to it's mode.
    for particle in particle_state:
        modes_state[int(particle)] += 1

    # numbers of particles in each mode
    return modes_state


def modes_state_to_particle_state(mode_state: np.ndarray, particles_number: int) -> np.ndarray:
    """
        Return given mode-basis state in particle basis.

        :param mode_state: Input state in mode-basis.
        :param particles_number: Number of particles.
        :return: Given mode-basis state in particle basis.
    """

    number_of_observed_modes = len(mode_state)
    modes = np.zeros(number_of_observed_modes)
    modes[:] = mode_state[:]
    particles_state = np.zeros(particles_number)

    i = k = 0
    while i < number_of_observed_modes:

        if modes[i] > 0:
            modes[i] -= 1
            particles_state[k] = int(i)
            k += 1
        else:
            i += 1

    return particles_state
