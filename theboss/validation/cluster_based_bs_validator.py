__author__ = "Tomasz Rybotycki"

"""
    This script contains an implementation of cluster based BS validator as presented
    in [7].
"""

from typing import Sequence, Tuple, List
from sklearn.cluster import KMeans
from math import ceil
from numpy import array


class ClusterBasedBSValidator:
    """
    This class implements the BS validator based on the clustering techniques (in this
    case only KMeans) as presented in the [7].
    """

    def __init__(self, modes_number: int) -> None:
        self._modes_number: int = modes_number

        # We use the middle of the rule of the thumb approach from [7] to select the
        # number of clusters.
        self._clusters_number: int = ceil(3 * modes_number / 4)

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

        # Note that e is computed differently than in [7].
        # TODO: Check with MO if that's correct.
        e: List[List[float]] = []

        for i in range(self._clusters_number):
            e.append(list())
            for j in range(len(samples_sizes)):
                e[i].append(
                    samples_sizes[j]
                    * trusted_sample_assignment.count(i)
                    / len(trusted_sample)
                )

        chi_squared: float = 0

        for i in range(self._clusters_number):
            for j in range(2):
                chi_squared += (
                    pow(samples_cluster_occupations[i][j] - e[i][j], 2) / e[i][j]
                )

        # TODO FROM HERE
        print(chi_squared)
        return False
