__author__ = "Tomasz Rybotycki"

from typing import List, Tuple, Sequence

from ..boson_sampling_utilities.boson_sampling_utilities import (
    mode_assignment_to_mode_occupation,
)
from ..boson_sampling_utilities.permanent_calculators.bs_permanent_calculator_interface import (
    BSPermanentCalculatorInterface,
)
from .generalized_cliffords_simulation_strategy import (
    GeneralizedCliffordsSimulationStrategy,
)


class ModeAssignmentGeneralizedCliffordsSimulationStrategyInterface(
    GeneralizedCliffordsSimulationStrategy
):
    """
    An implementation generalized C&C algorithm that works in the mode assignment
    description of the states (as the original C&C and the [2] description).
    """

    def __init__(self, bs_permanent_calculator: BSPermanentCalculatorInterface) -> None:
        super().__init__(bs_permanent_calculator)

    def simulate(
        self, input_state: Sequence[int], samples_number: int = 1
    ) -> List[Tuple[int, ...]]:
        """
            Returns sample from linear optics experiments given input state.

            :param input_state:
                Input state in the mode occupation representation.
            :param samples_number:
                Number of samples to simulate.

            :return:
                A list of sampled output states in the mode occupation description.
        """
        self.input_state = input_state
        self.number_of_input_photons = sum(input_state)
        self._get_sorted_possible_states()
        self.pmfs = dict()

        samples = []

        while len(samples) < samples_number:
            self._fill_r_sample()
            samples.append(
                mode_assignment_to_mode_occupation(self.r_sample, len(self.input_state))
            )
        return samples

    def _fill_r_sample(self) -> None:
        """
        Creates a sample according to the generalized C&C algorithm.
        """
        self.r_sample = tuple()
        self.current_key = self.r_sample
        self.current_sample_probability = 1

        while self.number_of_input_photons > len(self.r_sample):
            if self.current_key not in self.pmfs:
                self._calculate_new_layer_of_pmfs()
            self._sample_from_latest_pmf()

    def _calculate_new_layer_of_pmfs(self) -> None:
        """
        Adds new layer, from which new particle will be sampled, to the pmfs dict.
        """
        number_of_particle_to_sample = len(self.r_sample) + 1
        possible_input_states = self._labeled_states[number_of_particle_to_sample]
        corresponding_k_vectors = [
            [self.input_state[i] - state[i] for i in range(len(state))]
            for state in possible_input_states
        ]

        pmf = []

        weights = self._compute_weights_from_k_vectors(corresponding_k_vectors)
        weights /= sum(weights)
        self.possible_outputs[
            self.current_key
        ] = self._generate_possible_output_states()

        for output in self.possible_outputs[self.current_key]:
            output = mode_assignment_to_mode_occupation(output, len(self.input_state))
            pmf.append(0)
            for i in range(len(possible_input_states)):
                probability = self._calculate_outputs_probability(
                    possible_input_states[i], output
                )
                probability *= weights[i]
                pmf[-1] += probability

        self.pmfs[self.current_key] = pmf

    def _generate_possible_output_states(self) -> List[Tuple[int, ...]]:
        """
        Generates a list of possible output states in the current step of the algorithm
        basing on the current r_sample.

        :return:
            A list of the output state that one may get in the current algorithm's step.
        """
        possible_output_states: List[Tuple[int, ...]] = []

        for i in range(len(self.input_state)):
            possible_output_states.append(self.r_sample + (i,))

        return possible_output_states
