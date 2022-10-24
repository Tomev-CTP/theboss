__author__ = "Tomasz Rybotycki"
"""
    A script containing some utilities that could be used in general QC simulaitons.
"""

# TODO TR: Consider releasing this file as a separate package.

from typing import List, Dict, DefaultDict, Tuple, Set, Sequence

from numpy import (
    abs,
    linalg,
    log,
    sqrt,
    pi,
    exp,
    asarray,
    tile,
    power,
)
from collections import defaultdict


def count_total_variation_distance_dicts(
    distribution_1: Dict[Tuple[int, ...], float],
    distribution_2: Dict[Tuple[int, ...], float],
) -> float:
    """
    This method compute TVD between two distributions. We assume that both distributions
    sums up to 1.

    :param distribution_1:
        First distribution.
    :param distribution_2:
        Second distribution.

    :return:
        Total variation distance between the given distributions.
    """
    # Get common keys.
    keys: Set[Tuple[int, ...]] = set()
    keys.update(distribution_1.keys())
    keys.update(distribution_2.keys())

    # Wrap distributions into defaultdicts to take care of missing keys.
    distribution1: DefaultDict[Tuple[int, ...], float] = defaultdict(lambda: 0)
    distribution2: DefaultDict[Tuple[int, ...], float] = defaultdict(lambda: 0)

    distribution1.update(distribution_1)
    distribution2.update(distribution_2)

    # Compute the tvd.
    return 0.5 * sum([abs(distribution1[key] - distribution2[key]) for key in keys])


def count_total_variation_distance(
    distribution1: Sequence[float],
    distribution2: Sequence[float],
) -> float:
    """
    This method calculates total variation distance between two given distributions.

    :param distribution1:
        First distribution.
    :param distribution2:
        Second distribution.

    :return:
        Total variation distance between two given distributions.
    """

    assert len(distribution1) == len(distribution2), (
        " \
    f"
        f"Distributions must have equal lengths! Got: {len(distribution1)}"
        f"and {len(distribution2)}!"
    )
    total_variation_distance = 0

    for i in range(len(distribution1)):
        total_variation_distance += abs(distribution1[i] - distribution2[i])

    return total_variation_distance / 2


def count_distance_between_matrices(
    matrix1: Sequence[Sequence[complex]], matrix2: Sequence[Sequence[complex]]
) -> float:
    """
    Calculates distance between two given matrices. This method assumes, that the
    matrices have proper sizes.

    :param matrix1:
        First matrix.
    :param matrix2:
        Second matrix.

    :return:
        Distance between two given matrices.
    """
    return linalg.norm(asarray(matrix1) - asarray(matrix2))


def count_tv_distance_error_bound_of_experiment_results(
    outcomes_number: int, samples_number: int, error_probability: float
) -> float:
    """
    Calculates the distance bound between the experimental results and the n-sample
    estimation of these results.

    In case of large outcomes numbers one should consider solutions given here:
    https://math.stackexchange.com/questions/2696344/is-there-a-way-to-find-the-log-of-very-large-numbers

    In the method formally I should compute

        for prime_factor in prime_factors_of_the_large_number:
            error_bound += log(prime_factor)

    where the large_number =  2 ** outcomes_number - 2.

    However, by simply approximating large_number =  2 ** outcomes_number I can do the
    same with just

        error_bound += log(2) * outcomes_number

    without wasting a lot of time for calculating the prime factors or the number.

    :param outcomes_number:
        The number of possible outcomes that we consider.
    :param samples_number:
        Number of samples used for estimation.
    :param error_probability:
        Desired probability of error.

    :return:
        Bound on the tv distance between the estimate and the experimental results.
    """
    error_bound = -log(error_probability)
    error_bound += outcomes_number * log(2)  # APPROXIMATION!
    error_bound /= 2 * samples_number

    return sqrt(error_bound)


def get_prime_factors(number: int) -> List[int]:
    """
    For a given integer, returns its prime factors.

    :param number:
        Number of which prime factors will be returned.

    :return:
        Returns a list of prime factors of the number.
    """
    prime_factors = []

    while number % 2 == 0:
        prime_factors.append(2)
        number = number / 2

    for i in range(3, int(sqrt(number)) + 1, 2):
        while number % i == 0:
            prime_factors.append(i)
            number = number / i

    prime_factors.append(number)

    return prime_factors


def compute_minimal_number_of_samples_for_desired_accuracy(
    outcomes_number: int, error_probability: float, expected_distance: float
) -> int:
    """
    A function for computing the minimal number of samples for the approximate
    distribution (samples frequencies) to be in a specified TVD from the target
    distribution.

    :param outcomes_number:
        The number of outcomes in the approximated distribution.
    :param error_probability:
        The probability that the TVD would be greater than the expected distance.
    :param expected_distance:
        The TVD we expect the approximation to be from the exact distribution.

    :return:
        Returns the number of samples required to achieve given accuracy for
        specified parameters.
    """
    samples_number = -log(error_probability)

    samples_number += log(2) * outcomes_number

    samples_number /= 2 * pow(expected_distance, 2)

    return int(samples_number) + 1


def compute_qft_matrix(n: int) -> Sequence[Sequence[complex]]:
    """
    Computes n x n matrix of quantum fourier transform. The formula can be found
    e.g. on wiki

    https://en.wikipedia.org/wiki/Quantum_Fourier_transform

    :param n:
        Dimension of the array.

    :return:
        n x n ndarray of qft.
    """
    if n == 0:
        return asarray([])
    omega = exp(2j * pi / n)

    horizontal_range = tile(range(n), n).reshape(n, n)
    vertical_range = horizontal_range.transpose()
    full_range = horizontal_range * vertical_range
    qft_matrix = power(omega, full_range) / sqrt(n)

    return qft_matrix
