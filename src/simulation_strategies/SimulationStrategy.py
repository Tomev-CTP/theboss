__author__ = 'Tomasz Rybotycki'

import numpy as np


class SimulationStrategy:
    def simulate(self, input_state: np.ndarray):
        """
            Simulate the lossy boson sampling experiment.
            :param input_state: Input state of the simulation.
            :return:
        """
        raise NotImplementedError
