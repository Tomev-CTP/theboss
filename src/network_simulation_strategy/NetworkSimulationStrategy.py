from numpy import ndarray


class NetworkSimulationStrategy:
    def simulate(self, input_state: ndarray) -> ndarray:
        raise NotImplementedError
