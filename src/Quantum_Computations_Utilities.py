__author__ = 'Tomasz Rybotycki'

# TODO TR: Consider releasing this file as a separate package.

from typing import List

from numpy import abs, diag, dot, linalg, log2, ndarray, sqrt
from numpy.random import randn


def generate_haar_random_unitary_matrix(n: int) -> ndarray:
    """
        This method generates Haar random unitary n x n matrix. Using ideas from [3].
        :param n: Dimension of returned matrix.
        :return: Haar random unitary matrix m_u.
    """

    z = (randn(n, n) + 1j * randn(n, n))
    z /= sqrt(2.)
    q, r = linalg.qr(z)

    r = diag(r)
    lamb = diag(r / abs(r))

    m_u = dot(q, lamb)

    return m_u.T @ m_u


def count_total_variation_distance(distribution1: List[float], distribution2: List[float]) -> float:
    """
        This method calculates total variation distance between two given distributions.
        :param distribution1: First distribution.
        :param distribution2: Second distribution.
        :return: Total variation distance between two given distributions.
    """

    assert len(distribution1) == len(distribution2), "Distributions must be equi-length!"
    total_variation_distance = 0

    for i in range(len(distribution1)):
        total_variation_distance += abs(distribution1[i] - distribution2[i])

    return total_variation_distance / 2


def count_distance_between_matrices(matrix1: ndarray, matrix2: ndarray) -> float:
    """
        Calculates distance between two given matrices. This method assumes, that the matrices have proper sizes.
        :param matrix1: First matrix.
        :param matrix2: Second matrix.
        :return: Distance between two given matrices.
    """
    return linalg.norm(matrix1 - matrix2)


def count_tv_distance_error_bound_of_experiment_results(outcomes_number: int, samples_number: int,
                                                        error_probability: float):
    error_bound = log2(float(2 ** outcomes_number - 2)) - log2(error_probability)
    error_bound /= 2 * samples_number
    return sqrt(error_bound)
