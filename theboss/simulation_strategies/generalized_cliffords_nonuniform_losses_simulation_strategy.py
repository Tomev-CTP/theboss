__author__ = "Tomasz Rybotycki"

"""
A class for exact BS simulation using generalized C&C algorithm (version B).
"""

from numpy import delete, vstack, zeros_like
from typing import Sequence, Tuple

from theboss.simulation_strategies.generalized_cliffords_b_simulation_strategy import (
    GeneralizedCliffordsBSimulationStrategy,
    BSPermanentCalculatorInterface,
)

from theboss.simulation_strategies.simulation_strategy_interface import (
    SimulationStrategyInterface,
)

from theboss.boson_sampling_utilities import (
    prepare_interferometer_matrix_in_expanded_space,
)


class GeneralizedCliffordsNonuniformLossesSimulationStrategy(
    SimulationStrategyInterface
):
    """
    This class implements the generalized C&C algorithm for the optical networks with
    non-uniform (mode-dependent) losses. It utilizes the fact that we can interpret
    the losses as transferring the particle to an inaccessible mode in the expanded
    space. This, in turn, is done, by transforming the initial :math:`m \\cross m`
    matrix into a :math:`2m \\times 2m` matrix. For more details see and analyse the
    prepare_interferometer_matrix_in_expanded_space method.

    It expects the lossy interferometer matrix to be passed in bs_permanent_calculator.

    Note: Let :math:`l_i` denote the losses on the :math:`i`-th mode and
    :math:`t_i = 1 - l_i` denote transmission probability of the :math:`i`-th mode.
    Then, to apply losses to the lossless interferometer matrix one has to multiply it
    by a matrix with :math:`\\sqrt{t_i}` on diagonal.
    """

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        bs_permanent_calculator.matrix = (
            prepare_interferometer_matrix_in_expanded_space(
                bs_permanent_calculator.matrix
            )
        )

        # If for whatever reason one would like to run Clifford & Clifford A algorithm
        # for non-uniformly lossy networks using the expanded dimension approach, one
        # only has to change the helper strategy here.
        self._helper_strategy: GeneralizedCliffordsBSimulationStrategy = (
            GeneralizedCliffordsBSimulationStrategy(bs_permanent_calculator)
        )

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> [Tuple[int, ...]]:
        """
        The main method of the class. It returns desired number of samples from the
        (potentially non-uniformly lossy) BS experiment with given input state and
        interferometer matrix (specified previously).

        :param input_state:
            Input state of the BS experiment.
        :param samples_number:
            The number of samples to be returned.

        :return:
            Samples from the exact BS distribution.
        """
        expanded_state = vstack([input_state, zeros_like(input_state, dtype=int)])
        expanded_state = expanded_state.reshape(
            2 * len(input_state),
        )

        expanded_samples = self._helper_strategy.simulate(
            expanded_state, samples_number
        )
        # Trim the output states.
        samples = []

        for output_state in expanded_samples:
            while len(output_state) > len(input_state):
                output_state = delete(output_state, len(output_state) - 1)
            samples.append(tuple(output_state))

        return samples

    def set_new_matrix(self, matrix: Sequence[Sequence[complex]]) -> None:
        """
        Set the new interferometer matrix.

        :param matrix:
            New interferometer matrix.
        """
        self._helper_strategy.set_new_matrix(
            prepare_interferometer_matrix_in_expanded_space(matrix)
        )
