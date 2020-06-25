__author__ = 'Tomasz Rybotycki'
# TODO TR: Consider releasing this file as a separate package.

import numpy as np


def generate_haar_random_unitary_matrix(n: int) -> np.ndarray:
    """
        This method generates Haar random unitary n x n matrix. Using ideas from [3].
        :param n: Dimension of returned matrix.
        :return: Haar random unitary matrix m_u.
    """

    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n))
    z /= np.sqrt(2.)
    q, r = np.linalg.qr(z)

    r = np.diag(r)
    lamb = np.diag(r / np.abs(r))

    m_u = np.dot(q, lamb)

    return m_u.T @ m_u


def calculate_total_variation_distance(distribution1: list, distribution2: list) -> float:
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


def calculate_distance_between_matrices(matrix1: np.ndarray, matrix2: np.ndarray) -> float:
    """
        Calculates distance between two given matrices. This method assumes, that the matrices have proper sizes.
        :param matrix1: First matrix.
        :param matrix2: Second matrix.
        :return: Distance between two given matrices.
    """
    return np.linalg.norm(matrix1 - matrix2)
