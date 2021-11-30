__author__ = "Tomasz Rybotycki"

"""
    This file holds utility methods for Cliffords R implementation of their algorithm.
"""

from numpy import ndarray
from rpy2 import robjects


def numpy_array_to_r_matrix(numpy_array: ndarray) -> robjects.r.matrix:
    rows_number, columns_number = numpy_array.shape
    # Transposition is required as R inserts columns, not rows.
    r_values = robjects.ComplexVector([val for val in numpy_array.transpose().reshape(numpy_array.size)])
    return robjects.r.matrix(r_values, nrow=rows_number, ncol=columns_number)
