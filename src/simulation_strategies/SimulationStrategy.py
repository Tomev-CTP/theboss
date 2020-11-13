__author__ = 'Tomasz Rybotycki'

from numpy import ndarray


class SimulationStrategy:
    def simulate(self, input_state: ndarray) -> ndarray:
        """
            Simulate the lossy boson sampling experiment.
            :param input_state: Input state of the simulation.
            :return:
        """
        raise NotImplementedError
