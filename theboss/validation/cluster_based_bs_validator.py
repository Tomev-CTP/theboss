__author__ = "Tomasz Rybotycki"

"""
    This script contains an implementation of cluster based BS validator as presented
    in [7].
"""

from typing import Sequence, Tuple, List
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

    @property
    def modes_number(self) -> int:
        return self._modes_number

    @modes_number.setter
    def modes_number(self, modes_number: int) -> None:
        self._modes_number = modes_number

    def validate(
        self,
        trusted_sample: Sequence[Tuple[int, ...]],
        tested_sample: Sequence[Tuple[int, ...]],
    ) -> bool:
        k_means = KMeans(n_clusters=self._clusters_number, random_state=0)
        k_means.fit(array(trusted_sample))

        trusted_sample_assignment: List[int] = list(
            k_means.predict(array(trusted_sample))
        )

        tested_sample_assignment: List[int] = list(
            k_means.predict(array(tested_sample))
        )

        # Additional variable names are from [7].
        clusters_total_occupations: List[int] = []  # N_i
        samples_sizes: List[int] = [len(trusted_sample), len(tested_sample)]  # N_j

        samples_cluster_occupations: List[List[int]] = []  # N_ij

        for i in range(self._clusters_number):
            samples_cluster_occupations.append(list())
            samples_cluster_occupations[-1].append(trusted_sample_assignment.count(i))
            samples_cluster_occupations[-1].append(tested_sample_assignment.count(i))

        for i in range(self._clusters_number):
            clusters_total_occupations.append(
                trusted_sample_assignment.count(i) + tested_sample_assignment.count(i)
            )

        e: List[List[float]] = []

        for i in range(self._clusters_number):
            e.append(list())
            for j in range(len(samples_sizes)):
                e[i].append(
                    samples_sizes[j]
                    * clusters_total_occupations[i]
                    # / self._clusters_number  # As in [7]
                    / sum(samples_sizes)  # My hunch.
                )

        # Flatten the lists for the chi squared test
        observed: List[int] = []
        expected: List[float] = []

        for i in range(self._clusters_number):
            for j in range(2):
                observed.append(samples_cluster_occupations[i][j])
                expected.append(e[i][j])

        statistic, p_value = chisquare(observed, expected)
        print((statistic, p_value))
        return p_value > self._rejection_threshold

    def validate_majority_voting(
        self,
        trusted_sample: Sequence[Tuple[int, ...]],
        tested_sample: Sequence[Tuple[int, ...]],
    ) -> bool:
        """
        TODO if that's the one we will use.

        :param trusted_sample:
        :param tested_sample:
        :return:
        """
        repetitions: int = 11
        successes: int = 0

        for _ in range(repetitions):
            if self.validate(trusted_sample, tested_sample):
                successes += 1

        return successes > repetitions // 2

    def validate_tr(
        self,
        trusted_sample: Sequence[Tuple[int, ...]],
        tested_sample: Sequence[Tuple[int, ...]],
        random_seed: int = 0,
    ) -> bool:
        """
        TODO if that's the one that we will use in the end.

        :param trusted_sample:
        :param tested_sample:
        :return:
        """
        k_means = KMeans(n_clusters=self._clusters_number, random_state=random_seed)
        k_means.fit(array(trusted_sample))

        trusted_sample_assignment: List[int] = list(
            k_means.predict(array(trusted_sample))
        )

        tested_sample_assignment: List[int] = list(
            k_means.predict(array(tested_sample))
        )

        # Additional variable names are from [7].
        clusters_total_occupations: List[int] = []  # N_i
        samples_sizes: List[int] = [len(trusted_sample), len(tested_sample)]  # N_j

        samples_cluster_occupations: List[List[int]] = []  # N_ij

        for i in range(self._clusters_number):
            samples_cluster_occupations.append(list())
            samples_cluster_occupations[-1].append(trusted_sample_assignment.count(i))
            samples_cluster_occupations[-1].append(tested_sample_assignment.count(i))

        for i in range(self._clusters_number):
            clusters_total_occupations.append(
                trusted_sample_assignment.count(i) + tested_sample_assignment.count(i)
            )

        e: List[List[float]] = []

        for i in range(self._clusters_number):
            e.append(list())
            for j in range(len(samples_sizes)):
                e[i].append(
                    samples_sizes[j]
                    * samples_cluster_occupations[i][0]
                    / samples_sizes[0]
                )

        # Flatten the lists for the chi squared test
        observed: List[int] = []
        expected: List[float] = []

        for i in range(self._clusters_number):
            observed.append(samples_cluster_occupations[i][1])
            expected.append(e[i][1])

        statistic, p_value = chisquare(observed, expected)
        print((statistic, p_value))
        return p_value > self._rejection_threshold

    def validate_tr_majority_voting(
        self,
        trusted_sample: Sequence[Tuple[int, ...]],
        tested_sample: Sequence[Tuple[int, ...]],
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
            if self.validate_tr(trusted_sample, tested_sample, random_seed=i):
                successes += 1

        return successes > repetitions // 2
