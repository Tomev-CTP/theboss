__author__ = "Tomasz Rybotycki"

# TODO TR: Consider releasing this file as a separate package.

from typing import List, Union

import qutip
from numpy import abs, linalg, log2, ndarray, sqrt
from math import isinf


def generate_haar_random_unitary_matrix(d: int) -> ndarray:
    return qutip.rand_unitary_haar(d).full()


def count_total_variation_distance(distribution1: Union[List[float], ndarray],
                                   distribution2: Union[List[float], ndarray]) -> float:
    """
        This method calculates total variation distance between two given distributions.
        :param distribution1: First distribution.
        :param distribution2: Second distribution.
        :return: Total variation distance between two given distributions.
    """

    assert len(distribution1) == len(distribution2), f"Distributions must have equal lengths! Got: {len(distribution1)}" \
                                                     f"and {len(distribution2)}!"
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
                                                        error_probability: float) -> float:
    """
        Calculates the distance bound between the experimental results and the n-sample estimation of these results.

        In case of large outcomes numbers one should consider solutions given here:
        https://math.stackexchange.com/questions/2696344/is-there-a-way-to-find-the-log-of-very-large-numbers

        :param outcomes_number:
        :param samples_number: Number of samples used for estimation.
        :param error_probability: Desired probability of error.
        :return: Bound on the tv distance between the estimate and the experimental results.
    """
    possibly_huge_number = 2 ** outcomes_number - 2
    if not isinf(possibly_huge_number):
        prime_factors_of_the_number = get_prime_factors(possibly_huge_number)
    else:
        # In case the number is too big for python to process we need to approximate
        prime_factors_of_the_number = [2] * int(outcomes_number)

    error_bound = -log2(error_probability)

    for prime_factor in prime_factors_of_the_number:
        error_bound += log2(prime_factor)

    error_bound /= 2 * samples_number
    return sqrt(error_bound)


def get_prime_factors(number: int) -> List[int]:
    prime_factors = []

    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2

    for i in range(3,int(sqrt(number)) + 1,2):
        while number % i == 0:
            prime_factors.append(i)
            number = number / i

    prime_factors.append(number)

    return prime_factors


def compute_minimal_number_of_samples_for_desired_accuracy(outcomes_number: int, error_probability: float,
                                                           accuracy: float) -> int:

    possibly_huge_number = 2 ** outcomes_number - 2
    prime_factors_of_the_number = get_prime_factors(possibly_huge_number)

    samples_number = -log2(error_probability)

    for prime_factor in prime_factors_of_the_number:
        samples_number += log2(prime_factor)

    samples_number /= 2 * pow(accuracy, 2)

    return int(samples_number) + 1
