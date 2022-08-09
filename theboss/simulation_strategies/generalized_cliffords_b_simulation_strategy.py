__author__ = "Tomasz Rybotycki"

"""
    This script contains the implementation of the Generalized Cliffords version B
    algorithm. It differs from the original version in two places:
        -   In the k-th step, instead of computing k x k permanents, we compute a set of 
            k-1 x k-1 permanents. From that, we calculate the required k x k permanents
            in O(k) time each.
        -   We permute the columns of the matrix at the beginning. We can then sample
            from an easier distribution and obtain the same results.
"""

from typing import List, Sequence, Tuple

from numpy import zeros_like
from numpy.random import choice, randint

from theboss.simulation_strategies.generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
    BSPermanentCalculatorInterface,
)
from theboss.boson_sampling_utilities import mode_occupation_to_mode_assignment

from theboss.permanent_calculators.bs_cc_ryser_submatrices_permanent_calculator import (
    BSCCRyserSubmatricesPermanentCalculator,
)


class GeneralizedCliffordsBSimulationStrategy(GeneralizedCliffordsSimulationStrategy):
    """
    An implementation of generalized Clifford & Clifford strategy in its B version.
    """

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        super().__init__(bs_permanent_calculator)
        self._current_input = []
        self._working_input_state = None

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
        Returns sample from linear optics experiments given output state.

        :param input_state:
            Input state in particle basis.
        :param samples_number:
            Number of samples to simulate.

        :return:
            A list of samples from the interferometer.
        """
        self.input_state: Sequence[int] = input_state
        self.number_of_input_photons: int = sum(input_state)

        particle_input_state = list(mode_occupation_to_mode_assignment(input_state))

        samples = []

        while len(samples) < samples_number:
            self._current_input = zeros_like(input_state)
            self._working_input_state = particle_input_state.copy()
            self._fill_r_sample()
            samples.append(tuple(self.r_sample))
        return samples

    def _compute_pmf(self) -> None:

        self.pmf = []

        submatrices_permanents_calculator = BSCCRyserSubmatricesPermanentCalculator(
            self._bs_permanent_calculator.matrix, self._current_input, self.r_sample
        )

        submatrices_permanents = submatrices_permanents_calculator.compute_permanents()

        self._bs_permanent_calculator.input_state = self._current_input

        # New particle can come in any new mode
        for m in range(len(self.r_sample)):
            permanent = 0
            for i in range(len(self._current_input)):
                permanent_added = self._current_input[i] * submatrices_permanents[i]
                permanent_added *= self._bs_permanent_calculator.matrix[m][i]
                permanent += permanent_added

            self.pmf.append(abs(permanent) ** 2)

        total = sum(self.pmf)
        self.pmf = [val / total for val in self.pmf]

    def _fill_r_sample(self) -> None:
        self.r_sample = [0 for _ in self.input_state]

        while self.number_of_input_photons > sum(self.r_sample):
            self._update_current_input()
            self._compute_pmf()
            self._sample_from_pmf()

    def _update_current_input(self) -> None:
        self._current_input[
            self._working_input_state.pop(randint(0, len(self._working_input_state)))
        ] += 1

    def _sample_from_pmf(self) -> None:
        # TODO TR: Don't use numpy.random.choice, because it's slow.
        m = choice(range(len(self.input_state)), p=self.pmf)
        self.r_sample[m] += 1
