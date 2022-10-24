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
    case only k-means) as presented in the [7].
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
        """
        The total number of modes considered in the experiment that will be validated.
        It's used to determine the number of clusters, as proposed in [7].
        """
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
            A sample from bona-fide (trusted) boson sampler.
        :param seed:
            A seed for k-means initialization.
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
        The main method of the validator. It checks how well the clustering structure
        of ``tested_sample`` fits the clustering structure of bona fide BosonSampling
        used for the training.

        .. note::
            The validator should be trained before the validation.

        .. note::
            The authors of the method [7] advise to use majority voting.

        .. warning::
            The ``tr`` validation approach has very bad accuracy if ``tested_sample``
            comes from the bona fide BosonSampler.

        :param tested_sample:
            A sample from the tested BosonSampler.

        :param validation_approach:
            Can be ``original`` or ``tr``. The ``original`` is default and preferred
            option, since ``tr`` method is not giving satisfactory results.

        :return:
            ``True`` if the ``tested_sample`` seems to be obtained from the same
            distribution as the training sample. Else ``False``.
        """
        expected_values_function: Callable[
            [List[int], List[List[int]]], List[List[float]]
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
        Computes the expected values

        .. note::
            This method is slightly different from the one proposed by the authors in
            [7]. The denominator was fixed in such a way that the method started working
            as expected. Compare the code and [7] for details.

        :param samples_sizes:
            A list containing the size of training and tested sample, in that order.
        :param samples_cluster_occupations:
            Occupations of the respective clusters, in the cluster structure obtained
            during training, for both training and test samples, in that order.

        :return:
            The expected occupations of the clusters if the tested sample comes from the
            same distribution as the training sample.
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
        Computes the expected values for the ``tr`` validation approach. These expected
        values are basically scaled (according to the ``samples_sizes``) clusters
        occupations obtained during the training.

        :param samples_sizes:
            The sizes of both training and test samples, in that order.
        :param samples_cluster_occupations:
            Occupations of the respective clusters, in the cluster structure obtained
            during training, for both training and test samples, in that order.

        :return:
            The expected occupations of the clusters if the tested sample comes from the
            same distribution as the training sample.
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

    def validate_majority_voting(
        self,
        trusted_sample: Sequence[Tuple[int, ...]],
        tested_sample: Sequence[Tuple[int, ...]],
        validation_approach: str = "original",
        repetitions: int = 11,
    ) -> bool:
        """
        The preferred method of the validator. It runs the selected (via
        ``validation_approach``) validation method ``repetitions`` number of time
        and, as a result, returns the outcome that occurred the most.

        .. note::
            The ``repetitions=11`` is suggested by the authors of the method in [7]. In
            general, it should be an odd integer.

        :param trusted_sample:
            The sample from bona fide BosonSampler. It will be used for training.
        :param tested_sample:
            The sample from tested BosonSampler.
        :param validation_approach:
            The validation approach used by the validator. It can be either ``original``
            or ``tr``. The ``original`` one is preferred.
        :param repetitions:
            The number of the training repetitions.

        :return:
            ``True`` if both samples seem to come from the same distribution. Else
            ``False``.
        """
        successes: int = 0

        for i in range(repetitions):
            self.train(trusted_sample, i)
            if self.validate(tested_sample, validation_approach):
                successes += 1

        return successes > repetitions // 2
