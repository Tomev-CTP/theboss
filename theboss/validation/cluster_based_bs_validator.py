__author__ = "Tomasz Rybotycki"

"""
    This script contains an implementation of cluster based BS validator as presented
    in [7].
"""

from typing import Sequence, Tuple, List, Callable
from sklearn.cluster import KMeans
from math import ceil
from numpy import array
from scipy.stats import chisquare


class ClusterBasedBSValidator:
    """
    This class implements the BS validator based on the clustering techniques (in this
    case only KMeans) as presented in the [7].
    """

    def __init__(self, modes_number: int, rejection_threshold: float = 0.5) -> None:
        self._modes_number: int = modes_number

        # We use the middle of the rule of the thumb approach from [7] to select the
        # number of clusters.
        self._clusters_number: int = ceil(3 * modes_number / 4)
        self._rejection_threshold: float = rejection_threshold

        # Results of the validator "training"
        self._k_means: KMeans = KMeans(n_clusters=self._clusters_number, random_state=0)
        self._trusted_samples_assignment: List[int] = []
        self._trusted_sample_size: int = 0

    @property
    def modes_number(self) -> int:
        return self._modes_number

    @modes_number.setter
    def modes_number(self, modes_number: int) -> None:
        self._modes_number = modes_number

    def train(self, trusted_sample: Sequence[Tuple[int, ...]], seed: int = 0) -> None:
        """
        A method for validator training.

        This is basically the first step of the [7] algorithm which finds the clusters
        structure based on the trusted samples.

        :param trusted_sample:
            A sample from bona-fide boson sampler.
        :param seed:
            A seed for k means initialization.
        """
        self._k_means = KMeans(n_clusters=self._clusters_number, random_state=seed)
        self._k_means.fit(array(trusted_sample))

        self._trusted_samples_assignment = list(
            self._k_means.predict(array(trusted_sample))
        )

        self._trusted_sample_size = len(trusted_sample)

    def validate(
        self,
        tested_sample: Sequence[Tuple[int, ...]],
        validation_approach: str = "original",
    ) -> bool:
        """

        :param tested_sample:
        :param validation_approach:
            Can be "original" or "tr".
        :return:
        """
        expected_values_function: Callable[
            [List[int], List[List[int]]], List[List[int]]
        ]

        if validation_approach == "tr":
            expected_values_function = self._expected_values_tr
        else:
            expected_values_function = self._expected_values

        tested_sample_assignment: List[int] = list(
            self._k_means.predict(array(tested_sample))
        )

        # Additional variable names are from [7].
        clusters_total_occupations: List[int] = []  # N_i
        samples_sizes: List[int] = [
            self._trusted_sample_size,
            len(tested_sample),
        ]  # N_j

        samples_cluster_occupations: List[List[int]] = []  # N_ij

        for i in range(self._clusters_number):
            samples_cluster_occupations.append(list())
            samples_cluster_occupations[-1].append(
                self._trusted_samples_assignment.count(i)
            )
            samples_cluster_occupations[-1].append(tested_sample_assignment.count(i))

        e: List[List[float]] = expected_values_function(
            samples_sizes, samples_cluster_occupations
        )

        # Flatten the lists for the chi squared test
        observed: List[int] = []
        expected: List[float] = []

        for i in range(self._clusters_number):
            for j in range(2):
                observed.append(samples_cluster_occupations[i][j])
                expected.append(e[i][j])

        statistic, p_value = chisquare(observed, expected)
        # print((statistic, p_value))
        return p_value > self._rejection_threshold

    def _expected_values(
        self, samples_sizes: List[int], samples_cluster_occupations: List[List[int]]
    ) -> List[List[float]]:
        """

        :param samples_sizes:
        :param samples_cluster_occupations:
        :return:
        """
        e: List[List[float]] = []

        for i in range(self._clusters_number):
            e.append(list())
            for j in range(len(samples_sizes)):
                e[i].append(
                    samples_sizes[j]
                    * sum(samples_cluster_occupations[i])
                    # / self._clusters_number  # As in [7]
                    / sum(samples_sizes)  # My hunch.
                )

        return e

    def _expected_values_tr(
        self, samples_sizes: List[int], samples_cluster_occupations: List[List[int]]
    ) -> List[List[float]]:
        """

        :param samples_sizes:
        :param samples_cluster_occupations:
        :return:
        """
        e: List[List[float]] = []

        for i in range(self._clusters_number):
            e.append(list())
            for j in range(len(samples_sizes)):
                e[i].append(
                    samples_sizes[j]
                    * samples_cluster_occupations[i][0]
                    / samples_sizes[0]
                )

        return e

    def validate_tr(self, tested_sample: Sequence[Tuple[int, ...]],) -> bool:
        """
        TODO if that's the one that we will use in the end.

        :param trusted_sample:
        :param tested_sample:
        :return:
        """
        tested_sample_assignment: List[int] = list(
            self._k_means.predict(array(tested_sample))
        )

        # Additional variable names are from [7].
        clusters_total_occupations: List[int] = []  # N_i
        samples_sizes: List[int] = [
            self._trusted_sample_size,
            len(tested_sample),
        ]  # N_j

        samples_cluster_occupations: List[List[int]] = []  # N_ij

        for i in range(self._clusters_number):
            samples_cluster_occupations.append(list())
            samples_cluster_occupations[-1].append(
                self._trusted_samples_assignment.count(i)
            )
            samples_cluster_occupations[-1].append(tested_sample_assignment.count(i))

        for i in range(self._clusters_number):
            clusters_total_occupations.append(
                self._trusted_samples_assignment.count(i)
                + tested_sample_assignment.count(i)
            )

        e: List[List[float]] = self._expected_values_tr(
            samples_sizes, samples_cluster_occupations
        )

        # Flatten the lists for the chi squared test
        observed: List[int] = []
        expected: List[float] = []

        for i in range(self._clusters_number):
            observed.append(samples_cluster_occupations[i][1])
            expected.append(e[i][1])

        statistic, p_value = chisquare(observed, expected)
        # print((statistic, p_value))
        return p_value > self._rejection_threshold

    def validate_majority_voting(
        self,
        trusted_sample: Sequence[Tuple[int, ...]],
        tested_sample: Sequence[Tuple[int, ...]],
        validation_approach: str = "original",
    ) -> bool:
        """
        TODO if that's the one we will use.

        :param trusted_sample:
        :param tested_sample:
        :return:
        """
        repetitions: int = 11
        successes: int = 0

        for i in range(repetitions):
            self.train(trusted_sample, i)
            if self.validate(tested_sample, validation_approach):
                successes += 1

        return successes > repetitions // 2
