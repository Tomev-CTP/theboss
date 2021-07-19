__author__ = "Tomasz Rybotycki"

"""
    One should note, that R is required to use this module, as original Clifford's program is written in R. On my
    Windows 10, I am using anaconda and I had to add R_HOME env variable and R_path\bin, R_path\bin\x64 to the path. 
    https://cran.r-project.org/web/packages/BosonSampling/index.html
"""

from typing import List

from numpy import arange, array, array_split, int64, ndarray
from rpy2.robjects import packages

from .simulation_strategy_interface import SimulationStrategyInterface
from ..boson_sampling_utilities.boson_sampling_utilities import particle_state_to_modes_state
from ..rpy2_utilities import numpy_array_to_r_matrix


class CliffordsRSimulationStrategy(SimulationStrategyInterface):
    def __init__(self, interferometer_matrix: ndarray) -> None:
        self.interferometer_matrix = interferometer_matrix

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
        self.cliffords_r_sampler = boson_sampling_package.bosonSampler

    def set_matrix(self, interferometer_matrix: ndarray) -> None:
        self.interferometer_matrix = interferometer_matrix

    def simulate(self, initial_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        number_of_bosons = int(sum(initial_state))

        boson_sampler_input_matrix = numpy_array_to_r_matrix(self.interferometer_matrix[:, arange(number_of_bosons)])

        result, permanent, probability_mass_function = \
            self.cliffords_r_sampler(boson_sampler_input_matrix, sampleSize=samples_number, perm=False)

        # Add -1 to R indexation of modes (they start from 1).
        python_result = array([mode_value - 1 for mode_value in result], dtype=int64)
        samples_in_particle_states = array_split(python_result, number_of_bosons)

        samples = [particle_state_to_modes_state(sample, len(self.interferometer_matrix))
                   for sample in samples_in_particle_states]

        return samples
