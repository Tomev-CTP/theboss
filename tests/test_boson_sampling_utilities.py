__author__ = "Tomasz Rybotycki"

"""
    This script contains tests for the boson sampling utilities.
"""

import unittest
from theboss.boson_sampling_utilities import (
    mode_assignment_to_mode_occupation,
    mode_occupation_to_mode_assignment,
    bosonic_space_dimension,
    generate_possible_states,
    generate_lossy_n_particle_input_states,
    get_modes_transmissivity_values_from_matrix,
    prepare_interferometer_matrix_in_expanded_space,
    generate_state_types,
    compute_number_of_state_types,
    compute_number_of_states_of_given_type,
    compute_number_of_k_element_integer_partitions_of_n,
    generate_qft_matrix_for_first_m_modes,
    generate_random_phases_matrix_for_first_m_modes,
    generate_standard_state,
)

from theboss.quantum_computations_utilities import compute_qft_matrix

from typing import List, Tuple, Set, Iterable, Dict
from scipy.stats import unitary_group
from numpy import diag, sqrt, isclose, eye, nonzero, abs


class TestQuantumComputationsUtilities(unittest.TestCase):
    def setUp(self) -> None:
        """
        Basic method of the unittest.TestCase. Sets up the variables used in the tests.
        """

        # Corresponding states in mode occupation and mode assigment representations.
        self._mode_occupation_state: Tuple[int, ...] = (
            1,
            2,
            0,
            1,
            0,
        )
        self._mode_assigment_state: Tuple[int, ...] = (
            0,
            1,
            1,
            3,
        )
        self._trimmed_mode_occupation_state: Tuple[int, ...] = (
            1,
            2,
            0,
            1,
        )

        self._m: int = 10  # Modes number
        self._n: int = 5  # Particles number

        self._empty_state = tuple([0 for _ in range(self._m)])

        self._lossless_input: Tuple[int, ...] = (3, 2, 0, 1, 0)
        self._lossy_input_states: List[Tuple[int, ...]] = [
            (3, 2, 0, 1, 0),
            (3, 2, 0, 0, 0),
            (3, 1, 0, 1, 0),
            (3, 1, 0, 0, 0),
            (3, 0, 0, 1, 0),
            (3, 0, 0, 0, 0),
            (2, 2, 0, 1, 0),
            (2, 2, 0, 0, 0),
            (2, 1, 0, 1, 0),
            (2, 1, 0, 0, 0),
            (2, 0, 0, 1, 0),
            (2, 0, 0, 0, 0),
            (1, 2, 0, 1, 0),
            (1, 2, 0, 0, 0),
            (1, 1, 0, 1, 0),
            (1, 1, 0, 0, 0),
            (1, 0, 0, 1, 0),
            (1, 0, 0, 0, 0),
            (0, 2, 0, 1, 0),
            (0, 2, 0, 0, 0),
            (0, 1, 0, 1, 0),
            (0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0),
        ]

        # For the lossy matrix tests.
        self._transmissivities: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        self._matrix = unitary_group.rvs(len(self._transmissivities))
        self._matrix = self._matrix @ diag([sqrt(t) for t in self._transmissivities])

        # Matrix space expansion test.
        self._matrix_to_expand = eye(3) @ diag([sqrt(i / 10) for i in range(1, 4)])
        self._expanded_matrix = eye(6)

        for i in range(3):
            self._expanded_matrix[i][i] = sqrt(i / 10)
            self._expanded_matrix[i + 3][i + 3] = sqrt(i / 10)
            self._expanded_matrix[i + 3][i] = sqrt(1 - i / 10)
            self._expanded_matrix[i][i + 3] = sqrt(1 - i / 10)

        # QFT and random phases matrices generation tests.
        self._first_m_modes: int = 3
        self._all_modes: int = 3

    def test_mode_occupation_to_mode_assigment(self) -> None:
        """
        Check if a state in the mode occupation representation is properly
        transformed into a state in the mode assigment representation.
        """

        transformed_state: Tuple[int, ...] = mode_occupation_to_mode_assignment(
            self._mode_occupation_state
        )

        self.assertTrue(
            transformed_state == self._mode_assigment_state,
            f"{transformed_state} != {self._mode_assigment_state}",
        )

    def test_mode_assigment_to_mode_occupation(self) -> None:
        """
        Check if a state in the mode assigment representation is properly
        transformed into a state in the mode occupation representation.
        """

        transformed_state: Tuple[int, ...] = mode_assignment_to_mode_occupation(
            self._mode_assigment_state, len(self._mode_occupation_state)
        )

        self.assertTrue(
            transformed_state == self._mode_occupation_state,
            f"{transformed_state} != {self._mode_occupation_state}",
        )

    def test_mode_assigment_to_mode_occupation_wo_modes_number_specification(
        self,
    ) -> None:
        """
        Tests if the default behavior of the assigment to occupation representation,
        when the number of modes is not specified.
        """

        transformed_state: Tuple[int, ...] = mode_assignment_to_mode_occupation(
            self._mode_assigment_state
        )

        self.assertTrue(
            transformed_state == self._trimmed_mode_occupation_state,
            f"{transformed_state} != {self._trimmed_mode_occupation_state}",
        )

    def test_no_mode_states_generation(self) -> None:
        """
        Test if generate_possible_states method returns an empty list if the number of
        modes equals 0.
        """
        all_states: List[Tuple[int, ...]] = generate_possible_states(self._n, 0)
        self.assertTrue(len(all_states) == 0, f"There seems to be states with 0 modes!")

    def test_negative_particles_number_states_generation(self) -> None:
        """
        Test if generate_possible_states method returns an empty list if the number of
        particles is negative.
        """
        all_states: List[Tuple[int, ...]] = generate_possible_states(-1, self._m, True)
        self.assertTrue(
            len(all_states) == 0,
            f"There seems to be states with negative number of particles!",
        )

    def test_no_particles_states_generation(self) -> None:
        """
        Test if generate_possible_states method returns a particle-less state of proper
        size when the number of particles is equal to 0.
        """
        all_states = generate_possible_states(0, self._m, True)
        self.assertTrue(
            len(all_states) == 1, f"There's more than 1 particle-less state!"
        )
        self.assertTrue(
            all_states[0] == self._empty_state,
            f"{all_states[0]} != {self._empty_state}",
        )

    def test_generated_states_uniqueness(self) -> None:
        """
        Tests if every state generated by the generate_possible_states is unique.
        """
        all_states: List[Tuple[int, ...]] = generate_possible_states(
            self._n, self._m, True
        )

        self.assertTrue(
            len(set(all_states)) == len(all_states), f"Some states are not unique!"
        )

    def test_generated_states_modes_number(self) -> None:
        """
        Tests if every state has the proper modes number.
        """
        for state in generate_possible_states(self._n, self._m, True):
            self.assertTrue(
                len(state) == self._m,
                f"The number of modes in {state} is not {self._m}!",
            )

    def test_generated_states_particles_number(self) -> None:
        """
        Tests if every state has the proper particles number in a case without losses.
        """
        for state in generate_possible_states(self._n, self._m, False):
            self.assertTrue(
                sum(state) == self._n,
                f"The number of particles in {state} is not {self._n}!",
            )

    def test_generated_lossy_states_particles_number(self) -> None:
        """
        Tests if every state has equal or fewer particles than specified.
        """
        for state in generate_possible_states(self._n, self._m, True):
            self.assertTrue(
                sum(state) <= self._n,
                f"The number of particles in {state} is not <= {self._n}!",
            )

    def test_states_number(self) -> None:
        """
        Test if the number of generated states is theoretically correct.
        """

        all_states: List[Tuple[int, ...]] = generate_possible_states(self._n, self._m)

        theoretical_dimension: int = bosonic_space_dimension(self._n, self._m)

        self.assertTrue(
            len(all_states) == theoretical_dimension,
            f"{len(all_states)} != {theoretical_dimension}",
        )

    def test_lossy_states_number(self) -> None:
        """
        Test if the number of generated lossy states is theoretically correct.
        """

        all_lossy_states = generate_possible_states(self._n, self._m, True)

        theoretical_dimension: int = bosonic_space_dimension(self._n, self._m, True)

        self.assertTrue(
            len(all_lossy_states) == theoretical_dimension,
            f"{len(all_lossy_states)} != {theoretical_dimension}",
        )

    def test_lossy_input_states_generation(self) -> None:
        """
        Test if generate_lossy_input_states method actually generates all the lossy
        input states it should.
        """
        for particles_left_number in range(sum(self._lossless_input)):
            lossy_input_states: Set[Tuple[int, ...]] = set(
                generate_lossy_n_particle_input_states(
                    self._lossless_input, particles_left_number
                )
            )
            for state in self._lossy_input_states:
                if sum(state) == particles_left_number:
                    self.assertTrue(
                        state in lossy_input_states,
                        f"{state} is not in the lossy input states {lossy_input_states} of {self._lossless_input} for n = {particles_left_number}!",
                    )

    def test_modes_transmissivity_value_computation(self) -> None:
        """
        Test if proper transmissivities are obtained from the lossy matrix.

        Notice that due to approximation this might not be accurate, hence the isclose
        method is used for comparison.
        """
        transmissivities: Set[float] = set(
            get_modes_transmissivity_values_from_matrix(self._matrix)
        )
        for transmissivity in self._transmissivities:
            self.assertTrue(
                self._is_close_to_any(transmissivity, transmissivities),
                f"{transmissivity} not in {transmissivities}!",
            )

    @staticmethod
    def _is_close_to_any(value: float, values: Iterable[float]) -> bool:
        """
        This method checks if value is close to any of given values.

        :param value:
            Value to compare with the other values.

        :param values:
            Other values used for the comparison.

        :return:
            True is value is close to any of the values, else False.
        """
        for v in values:
            if isclose(value, v):
                return True
        return False

    def test_expanded_space_lossy_interferometer_preparation(self) -> None:
        """
        Tests if the lossy interferometer is properly expanded into the higher dimension
        for nonuniform losses' simulation.
        """
        expanded_interferometer = prepare_interferometer_matrix_in_expanded_space(
            self._matrix_to_expand
        )
        self.assertTrue((expanded_interferometer == self._expanded_matrix).all)

    def test_expanded_space_lossless_interferometer_preparation(self) -> None:
        """
        Tests if the lossless interferometer is properly expanded into the higher
        dimension (as an edge-case).
        """
        m: int = 3
        expanded_matrix = prepare_interferometer_matrix_in_expanded_space(eye(m))
        self.assertTrue((expanded_matrix == eye(2 * m)).all)

    def test_lossless_state_types_generation(self) -> None:
        """
        Tests if all lossless state types are generated.
        """
        self._test_state_types_generation(losses=False)

    def test_lossy_state_types_generation(self) -> None:
        """
        Tests if all lossy state types are generated.
        """
        self._test_state_types_generation(losses=True)

    def _test_state_types_generation(self, losses: bool = False) -> None:
        """
        This method takes care of boilerplate code concerning state types generation.

        :param losses:
            A flag informing if losses should be considered.
        """
        all_states: List[Tuple[int, ...]] = generate_possible_states(
            self._n, self._m, losses
        )
        state_types: Set[Tuple[int, ...]] = set()
        generated_state_types = generate_state_types(self._m, self._n, losses)

        for state in all_states:
            state_types.add(tuple(sorted(state, reverse=True)))

        self.assertTrue(len(generated_state_types) == len(state_types))

    def test_lossless_state_types_number_computation(self) -> None:
        """
        Test if the number of lossless state types is computed properly.
        """
        self._test_state_types_number_computation()

    def test_lossy_state_types_number_computation(self) -> None:
        """
        Test if the number of lossy state types is computed properly.
        """
        self._test_state_types_number_computation(True)

    def _test_state_types_number_computation(self, losses: bool = False) -> None:
        """
        This method takes care of boilerplate code concerning state types number
        computation.

        :param losses:
            A flag informing if losses should be considered.
        """
        state_types_number = compute_number_of_state_types(self._m, self._n, losses)
        state_types = generate_state_types(self._m, self._n, losses)

        self.assertTrue(
            len(state_types) == state_types_number,
            f"State types number is different than {state_types_number}!",
        )

    def test_number_of_states_of_given_type_computation(self) -> None:
        """
        Test if the number of states of given type are properly computed.
        """
        state_types: List[Tuple[int, ...]] = generate_state_types(self._m, self._n)
        all_states: List[Tuple[int, ...]] = generate_possible_states(self._n, self._m)

        counts: Dict[Tuple[int, ...], int] = {}

        for state_type in state_types:
            counts[state_type] = 0

        for state in all_states:
            counts[tuple(sorted(state, reverse=True))] += 1

        for state_type in state_types:
            states_of_given_type_number = compute_number_of_states_of_given_type(
                state_type
            )
            self.assertTrue(
                states_of_given_type_number == counts[state_type],
                f"{states_of_given_type_number} != {counts[state_type]}",
            )

    def test_number_of_k_element_partitions_of_n_computation(self) -> None:
        """
        Test if the number of :math:`k`-element partitions of integer :math:`n` is
        computed properly. Notice that the number of partitions is closely related to
        the number of state types, thus the latter can be used in the tests.
        """
        state_types: List[Tuple[int, ...]] = generate_state_types(self._n, self._n)
        counts: Dict[int, int] = {}

        for i in range(self._n):
            counts[i + 1] = 0

        for state_type in state_types:
            counts[len(nonzero(state_type)[0])] += 1

        for k in range(1, self._n + 1):
            self.assertTrue(
                counts[k]
                == compute_number_of_k_element_integer_partitions_of_n(k, self._n),
                f"{counts[k]} != {compute_number_of_k_element_integer_partitions_of_n(k, self._n)}",
            )

    def test_qft_matrix_generation_on_first_m_modes(self) -> None:
        """
        Test if the :math:`m \\times m` QFT matrix is properly embedded into the first
        :math:`m` modes of a bigger (interferometer) matrix.
        """

        # This should be tested in the other file.
        small_qft_matrix = compute_qft_matrix(self._first_m_modes)
        test_matrix = eye(self._all_modes, dtype=complex)

        for i in range(self._first_m_modes):
            for j in range(self._first_m_modes):
                test_matrix[i][j] = small_qft_matrix[i][j]

        generated_qft_matrix = generate_qft_matrix_for_first_m_modes(
            self._first_m_modes, self._all_modes
        )

        self.assertTrue((test_matrix == generated_qft_matrix).all)

    def test_random_phases_matrix_on_first_m_modes(self) -> None:
        """
        Test if the :math:`m \\times m` random phases diagonal matrix is properly
        embedded into the first :math:`m` modes of a bigger (interferometer) matrix.
        """
        generated_matrix = generate_random_phases_matrix_for_first_m_modes(
            self._first_m_modes, self._all_modes
        )

        test_matrix = eye(self._all_modes, dtype=complex)

        # Add random phases to the test matrix.
        for i in range(self._first_m_modes):
            test_matrix[i][i] = generated_matrix[i][i]
            # Also test if random number are actually phases.
            self.assertTrue(
                isclose(abs(test_matrix[i][i]), 1),
                f"|{test_matrix[i][i]}| = {abs(test_matrix[i][i])} != 1",
            )

        self.assertTrue((generated_matrix == test_matrix).all)

    def test_standard_input_state_generation(self) -> None:
        """
        Tests if standard input state is generated correctly.
        """
        correct_state: List[int] = [0 for _ in range(self._m)]
        correct_state[0 : self._n] = [1 for _ in range(self._n)]
        self.assertTrue(
            generate_standard_state(self._m, self._n) == tuple(correct_state)
        )
