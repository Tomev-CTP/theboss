from rpy2 import robjects
from numpy import array, complex


def numpy_array_to_r_matrix(numpy_array: array) -> robjects.r.matrix:
    rows_number = len(numpy_array)
    columns_number = len(numpy_array[0])
    r_matrix_range = rows_number * columns_number

    r_matrix = robjects.r.matrix(robjects.ComplexVector([complex(i) for i in range(r_matrix_range)]), nrow=rows_number)

    for col_num in range(columns_number):
        for row_num in range(rows_number):
            r_matrix[int(row_num + col_num * rows_number)] = complex(numpy_array[row_num][col_num])

    return r_matrix
