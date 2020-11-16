__author__ = 'Tomasz Rybotycki'

from typing import List

from numpy import ndarray


class SimulationStrategy:
    def simulate(self, input_state: ndarray, samples_number: int = 1) -> List[ndarray]:
        """
            Simulate the lossy boson sampling experiment.

            :param input_state: Input state of the simulation.
            :param samples_number: Number of samples one wants to simulate.
            :return:
        """
        raise NotImplementedError
