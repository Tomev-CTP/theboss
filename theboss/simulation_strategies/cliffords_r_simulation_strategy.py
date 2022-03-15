__author__ = "Tomasz Rybotycki"

"""
    One should note, that R is required to use this module, as original Clifford's
    program is written in R. On my Windows 10, I am using anaconda and I had to add
    R_HOME env variable and R_path\bin, R_path\bin\x64 to the path. 
    https://cran.r-project.org/web/packages/BosonSampling/index.html
"""

from numpy import arange, array, array_split, int64, ndarray

from .simulation_strategy_interface import SimulationStrategyInterface

try:
    from rpy2 import robjects
    from rpy2.robjects import packages

    from typing import List, Dict, Tuple

    from ..boson_sampling_utilities.boson_sampling_utilities import \
        particle_state_to_modes_state


    class CliffordsRSimulationStrategy(SimulationStrategyInterface):
        def __init__(self, interferometer_matrix: ndarray) -> None:
            self.interferometer_matrix = interferometer_matrix

            required_packages = ('BosonSampling', 'Rcpp', 'RcppArmadillo')

            # Check if required R packages are installed. Inform if not.
            if not all(packages.isinstalled(package) for package in required_packages):
                print('Some packages are missing! Missing packages:')
                for package in required_packages:
                    if not packages.isinstalled(package):
                        print(package)

            boson_sampling_package = packages.importr('BosonSampling')
            self.cliffords_r_sampler = boson_sampling_package.bosonSampler

        def set_matrix(self, interferometer_matrix: ndarray) -> None:
            self.interferometer_matrix = interferometer_matrix

        @staticmethod
        def _numpy_array_to_r_matrix(numpy_array: ndarray) -> robjects.r.matrix:
            rows_number, columns_number = numpy_array.shape
            # Transposition is required as R inserts columns, not rows.
            r_values = robjects.ComplexVector(
                [val for val in numpy_array.transpose().reshape(numpy_array.size)])
            return robjects.r.matrix(r_values, nrow=rows_number, ncol=columns_number)

        def simulate(self, initial_state: ndarray, samples_number: int = 1) -> List[
            ndarray]:
            """
                Simulate BS experiment for given input.

                Note:   The results of Clifford & Clifford method are given in the first
                        quantization description (mode assignment)!

                :param initial_state:   Input state in the modes occupation description.
                :param samples_number:  Number of samples to sample.

                :return:    List of samples in the first quantization description (mode
                            assignment)
            """
            number_of_bosons = int(sum(initial_state))

            boson_sampler_input_matrix = self._numpy_array_to_r_matrix(
                self.interferometer_matrix[:, arange(number_of_bosons)])

            result, permanent, probability_mass_function = \
                self.cliffords_r_sampler(boson_sampler_input_matrix,
                                         sampleSize=samples_number,
                                         perm=False)

            # Add -1 to R indexation of modes (they start from 1).
            python_result = array([mode_value - 1 for mode_value in result],
                                  dtype=int64)
            samples_in_particle_states = array_split(python_result, samples_number)

            # There are some problems with the actual and theoretical runtimes. The
            # reason for that could be parsing the result to a second quantization
            # description.
            # return samples_in_particle_states
            samples_in_occupation_description = []

            for sample in samples_in_particle_states:
                samples_in_occupation_description.append(
                    particle_state_to_modes_state(sample, len(initial_state)))

            return samples_in_occupation_description

        def find_probabilities(self, initial_state: ndarray,
                               outcomes_of_interest: List[ndarray]) \
                -> Dict[Tuple[int, ...], float]:

            number_of_bosons = int(sum(initial_state))

            outcomes_of_interest = [tuple(o) for o in outcomes_of_interest]

            outcomes_probabilities: dict = {}

            boson_sampler_input_matrix = self._numpy_array_to_r_matrix(
                self.interferometer_matrix[:, arange(number_of_bosons)])

            while len(outcomes_probabilities) != len(outcomes_of_interest):

                result, permanent, pmf = \
                    self.cliffords_r_sampler(boson_sampler_input_matrix,
                                             sampleSize=1,
                                             perm=True)

                # Add -1 to R indexation of modes (they start from 1).
                python_result = array([mode_value - 1 for mode_value in result],
                                      dtype=int64)
                samples_in_particle_states = array_split(python_result, 1)
                print(samples_in_particle_states)

                sample = tuple(particle_state_to_modes_state(samples_in_particle_states,
                                                             len(initial_state)))

                if sample in outcomes_of_interest:
                    outcomes_probabilities[sample] = pmf[0]
                else:
                    print(f"\t{sample} not in {outcomes_of_interest}")

                print(f"{len(outcomes_probabilities)} / {len(outcomes_of_interest)}")

            return outcomes_probabilities

except ImportError as e:

    print(f"An import error occurred during Cliffords R Strategy initialization:\n\t{e}")

    class CliffordsRSimulationStrategy(SimulationStrategyInterface):
        def __init__(self, interferometer_matrix: ndarray) -> None:
            print("You have to install rpy2 package in order to use CliffordsRSimulationStrategy.")
            raise NotImplementedError

        def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
            raise NotImplementedError
