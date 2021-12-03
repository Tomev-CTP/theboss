__author__ = "Tomasz Rybotycki"

import unittest

from numpy import conjugate, identity, ndarray, transpose, allclose

from ..BoSS_Tomev.quantum_computations_utilities import generate_haar_random_unitary_matrix


class TestQuantumComputationsUtilities(unittest.TestCase):

    def setUp(self) -> None:
        self.matrix_size = 5
        self.number_of_matrices_for_distinct_elements_check = 10  # Should be >= 2.
        pass

    def test_unitarity_of_matrices_generated_by_haar_random_unitary_method(self) -> None:
        random_matrix = generate_haar_random_unitary_matrix(self.matrix_size)
        random_matrix_hermitian_adjoint = transpose(conjugate(random_matrix))
        product_of_matrix_and_hermitian_adjoint = random_matrix_hermitian_adjoint.dot(random_matrix)

        identity_matrix = identity(self.matrix_size, dtype=complex)

        self.assertTrue(self.__are_matrices_elementwise_close(identity_matrix, product_of_matrix_and_hermitian_adjoint))

    @staticmethod
    def __are_matrices_elementwise_close(matrix1: ndarray, matrix2: ndarray) -> bool:
        #  I assume that there are only rectangular matrices
        if len(matrix2) != len(matrix1):
            return False
        if len(matrix2[0]) != len(matrix1[0]):
            return False

        return allclose(matrix1, matrix2)

    def test_haar_random_unitary_matrices_generation_differences(self) -> None:
        generated_unitaries_matrices = \
            [generate_haar_random_unitary_matrix(self.matrix_size) for _ in
             range(self.number_of_matrices_for_distinct_elements_check)]

        are_all_matrices_different = []

        for i in range(self.number_of_matrices_for_distinct_elements_check):
            for j in range(i + 1, self.number_of_matrices_for_distinct_elements_check):
                are_all_matrices_different.append(self.__are_matrices_elementwise_close(generated_unitaries_matrices[i],
                                                                                        generated_unitaries_matrices[j])
                                                  )
        self.assertTrue(not any(are_all_matrices_different))
