__author__ = 'Tomasz Rybotycki'
# One should note, that R is required to use this module, as original Clifford's program is written in R.
# On my Windows 10, I am using anaconda and I had to add R_HOME env variable and R_path\bin, R_path\bin\x64 to the path.

from rpy2.robjects import packages
from rpy2 import robjects
from numpy import ndarray, zeros, array

required_packages = ('BosonSampling', 'Rcpp', 'RcppArmadillo')

# Check if required R packages are installed. And note if not.
if all(packages.isinstalled(package) for package in required_packages):
    print('All R packages are installed!')
else:
    print('Some packages are missing! Missing packages:')
    for package in required_packages:
        if not packages.isinstalled(package):
            print(package)

boson_sampling_package = packages.importr('BosonSampling')
#print(boson_sampling_package.__dict__['_rpy2r'])
unitary_matrix_generator = boson_sampling_package.randomUnitary
boson_sampler = boson_sampling_package.bosonSampler

given_matrix = array([
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ])

n = 3
number_of_samples = 100
first_n_columns_of_given_matrix = given_matrix[:, [i for i in range(n)]]


def numpy_array_to_r_matrix(numpy_array: array) -> robjects.r.matrix:
    rows_number = len(numpy_array)
    columns_number = len(numpy_array[0])
    r_matrix_range = rows_number * columns_number
    r_matrix = robjects.r.matrix(robjects.FloatVector(range(r_matrix_range)), nrow=rows_number)

    for col_num in range(columns_number):
        for row_num in range(rows_number):
            r_matrix[int(row_num + col_num * rows_number)] = float(numpy_array[row_num][col_num])

    return r_matrix


boson_sampler_input_matrix = numpy_array_to_r_matrix(first_n_columns_of_given_matrix)
vals, perms, pmfs = boson_sampler(boson_sampler_input_matrix, 1, True)

print(vals)
