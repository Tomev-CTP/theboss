__author__ = 'Tomasz Rybotycki'

from numpy import ndarray
from typing import List


class SimulationStrategy:
    def simulate(self, input_state: ndarray) -> List[int]:
        """
            Simulate the lossy boson sampling experiment.
            :param input_state: Input state of the simulation.
            :return:
        """
        raise NotImplementedError
